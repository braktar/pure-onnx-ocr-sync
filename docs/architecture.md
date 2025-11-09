# アーキテクチャ設計書：Pure Rust OnnxOCR

作成者: Shion Watanabe  
日付: 2025-11-09  
リポジトリ: http://github.com/siska-tech/pure-onnx-ocr

## 🎯 目的

  * **プログラム全体の構造（モジュール構成）を決定する。**
    システムは、推論の核となる `OcrEngine` を中心に配置し、その周囲に「検出パイプライン」と「認識パイプライン」を独立したモジュールとして配置する。各パイプラインは、さらに「前処理」「後処理」のサブモジュールに分割される。

  * **主要なモジュール間の役割分担と依存関係を明確にする。**
    `OcrEngine` が全体の制御フロー（オーケストレーション）を担当する。`DetectionModule` は画像からポリゴン（座標）を生成する責務を持ち、`RecognitionModule` はポリゴンと元画像からテキストを生成する責務を持つ。モジュールは互いに疎結合であり、例えば `DetectionModule` は `RecognitionModule` の存在を知る必要はない。

  * **採用する設計原則やデザインパターンを定義する。**
    「関心の分離 (Separation of Concerns)」を最重要の原則とする。特に、ONNXモデルの推論、画像処理、ジオメトリ計算、CTCデコードなど、専門性の高い領域を明確にモジュール化する。外部APIとしては「Facade パターン」を採用し、`OcrEngine` が内部の複雑な処理をカプセル化し、利用者にシンプルなインターフェースを提供する。

-----

## 記載すべき項目

### 1\. システム構成図（コンポーネント図）

プログラムは、以下の主要モジュールによって構成されます。

``` mermaid
flowchart TD
    %% --- 利用者層 ---
    A["利用者 (Application)"] --> B["Public API (Facade)<br>OcrEngine<br><br>- det_model: TractModel<br>- rec_model: TractModel<br>- dictionary: Vec<String><br>- config: OcrConfig<br><br>+ new(...) -> Self<br>+ run(image_path) -> Vec<OcrResult>"]

    %% --- 検出と認識の分岐 ---
    B --> C1["テキスト領域検出<br>責務: 画像からテキスト領域(ポリゴン)を発見"]
    B --> C2["文字認識<br>責務: 画像領域からテキストを解読"]

    %% --- 前処理ステージ ---
    C1 --> D1[画像前処理<br>- image::resize<br>- ndarray::permuted_axes]
    C2 --> D2["画像前処理<br>- image::crop<br>- image::resize_exact<br>- ndarray::stack (Batching)"]

    %% --- 推論ステージ ---
    D1 --> E1["Inference Module<br>(det.onnx)<br>- tract_onnx::run()"]
    D2 --> E2["Inference Module<br>(rec.onnx)<br>- tract_onnx::run()"]

    %% --- 後処理ステージ ---
    E1 --> F1[後処理<br>- imageproc::contours<br>- i_overlay::buffering<br>- geo_types::Polygon]
    E2 --> F2[後処理<br>- ndarray::argmax_axis<br>- Custom CTC Greedy Decode<br>- Dictionary Lookup]
```

### 2\. モジュール間の関係

#### 2.1. データフロー

全体のデータフローは、`OcrEngine::run()` メソッドの呼び出しを起点とします。

1.  **入力:** 利用者が `OcrEngine::run()` に画像パス（`&str`）を提供します。
2.  **画像ロード:** `image` クレートが画像を `DynamicImage` としてロードします。
3.  **検出 (Detection):**
    a.  `DetPreProcessor` が `DynamicImage` を受け取り、リサイズ（アスペクト比維持）、正規化、NCHW形式への軸転置 [1, 2, 3] を行い、`tract::Tensor`（検出用入力）を生成します。
    b.  `Inference Module` が `det.onnx` を実行し、`tract::Tensor`（確率マップ）を出力します。
    c.  `DetPostProcessor` が確率マップを受け取り、`imageproc::contours` [4, 5] で輪郭を抽出し、`i_overlay::buffering` [6] でポリゴンを拡大（オフセット）し、`Vec<geo_types::Polygon>` [7] を生成します。
4.  **認識 (Recognition):**
    a.  `RecPreProcessor` が `Vec<Polygon>` と元の `DynamicImage` を受け取ります。
    b.  ポリゴン毎に画像をクロップし、固定サイズ（例：\`\` [8, 9]）に強制リサイズ [10] し、正規化します。
    c.  `ndarray::stack` [11] を使い、複数のクロップ画像を単一のバッチ \`tract::Tensor\`（認識用入力）にまとめます。
    d.  \`Inference Module\` が \`rec.onnx\` を実行し、\`tract::Tensor\`（クラスロジット）を出力します。
    e.  \`RecPostProcessor\` がロジットを受け取り、\`ndarray::argmax\_axis\` [12] で各タイムステップの最大インデックスを取得します。
    f.  Pure Rustで実装されたCTC Greedyデコードロジック [13, 14]（重複とブランクID [15, 16] の削除）を実行します。
    g.  \`OcrEngine\` が保持する辞書（\`Vec\<String\>\`）でインデックスを \`String\` にマッピングします。
5.  **出力:** `DetPostProcessor` からの `Vec<Polygon>` と `RecPostProcessor` からの `Vec<String>` を集約し、最終的な `Vec<OcrResult>` [17] を利用者に返します。

#### 2.2. 処理シーケンス（主要ユースケース: `OcrEngine::run`）

`OcrEngine::run` が呼び出された際の、モジュール間の主要なインタラクションは以下の通りです。

1.  `App -> OcrEngine.run(path)`
2.  `OcrEngine -> image::open(path)`
3.  `OcrEngine -> DetPreProcessor.process(image, config.det_limit_side_len)`
4.  `OcrEngine -> InferenceModule(det_model).run(det_input)`
5.  `OcrEngine -> DetPostProcessor.process(det_output, config.det_unclip_ratio)`
6.  `OcrEngine -> RecPreProcessor.process(image, polygons, config.rec_image_shape)`
7.  `OcrEngine -> InferenceModule(rec_model).run(rec_batch)`
8.  `OcrEngine -> RecPostProcessor.decode(rec_output, engine.dictionary, engine.blank_id)`
9.  `OcrEngine -> App.return(Vec<OcrResult>)`

*注：* 複数のテキスト領域が検出された場合、ステップ 6〜8 はバッチ処理（または `rayon` [11] による並列イテレーション）として実行されます。

### 3\. 設計原則・デザインパターン

  * **設計原則: 関心の分離 (Separation of Concerns)**
    本アーキテクチャの核心原則です。「Pure Rust」要件に基づき、C++ライブラリが担っていた各責務を、対応するPure Rustクレートに置き換えます。

      * **推論:** `onnxruntime` [18] -\> `tract-onnx` [18]
      * **画像処理:** `opencv-python` [19] -\> `image` + `imageproc` [20]
      * **ジオメトリ:** `pyclipper` [21], `shapely` [22] -\> `i_overlay` [6], `geo-types` [23]
      * **N次元配列:** `numpy` [24] -\> `ndarray` [25]

  * **設計原則: カプセル化 (Encapsulation)**
    `OcrEngine` 構造体が、OCRパイプラインのすべての複雑な状態（ロードされたモデル、辞書、設定）とロジック（前処理、後処理の呼び出し）をカプセル化します。利用者は `run()` を呼び出すだけでよく、内部の2段階プロセス（検出と認識）を意識する必要はありません。

  * **デザインパターン: Facade パターン**
    `OcrEngine` は、`DetPreProcessor`, `DetPostProcessor`, `RecPreProcessor`, `RecPostProcessor`, `InferenceModule` という多数のサブシステムに対する統一された高レベルインターフェース（Facade）として機能します。

  * **デザインパターン: Builder パターン（推奨）**
    `OcrEngine` の初期化は、検出モデルパス、認識モデルパス、辞書パス、各種設定（`OcrConfig`）など、多くのパラメータを必要とします。将来的な拡張性を考慮し、`OcrEngineBuilder` を実装して、安全かつ柔軟なインスタンス構築を可能にすることを推奨します。

### 4\. 技術選定

要件定義書の「Pure Rust」制約に基づき、C/C++ FFIに依存するライブラリを排除し、以下のPure Rustクレートを選定します。

| カテゴリ           | 選定クレート      | 選定理由（Pure Rust代替）                                                                                                                    |
| :----------------- | :---------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **ONNX推論**       | `tract-onnx` [26] | `onnxruntime-rs` [27, 18] のPure Rust代替。C++ランタイムへの依存を排除する [18] ための**必須**の選択。                                       |
| **N次元配列**      | `ndarray` [28]    | `numpy` [25, 24] のPure Rust代替。テンソルの正規化、軸転置（HWC-\>NCHW）[1, 3]、`argmax` [12] 処理に使用。                                   |
| **画像I/O・処理**  | `image` [29]      | `opencv-python` [20, 19] のPure Rust代替。`imread`, `resize` (`resize_exact`) [10], `crop` に使用。                                          |
| **輪郭検出**       | `imageproc` [30]  | `cv2.findContours` [31, 32, 33] のPure Rust代替 [4, 5]。DBNet後処理の核となる輪郭抽出に使用。                                                |
| **ポリゴン処理**   | `i_overlay` [6]   | `pyclipper` [34, 21, 35]（C++ラッパー）のPure Rust代替 [36, 37]。DBNet後処理のポリゴン拡大（Buffering）[6] に使用。                          |
| **ジオメトリ表現** | `geo-types` [23]  | `shapely` [22]（GEOS C APIラッパー）のPure Rust代替 [38, 39]。ポリゴンや座標の標準的なデータ構造として使用 [7]。                             |
| **並列処理**       | `rayon` [11]      | （オプション）認識パイプライン（`RecPreProcessor` -\> `RecPostProcessor`）をテキスト領域ごとに並列化し、スループットを向上させるために使用。 |