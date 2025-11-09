# `pure-onnx-ocr` (Pure Rust OnnxOCR)

[](https://www.google.com/search?q=https://crates.io/crates/pure_onnx_ocr)
[](https://www.google.com/search?q=https://docs.rs/pure_onnx_ocr)
([https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg])

`jingsongliujing/OnnxOCR` [1] プロジェクトのOCRパイプラインを、C/C++依存なしの **Pure Rust** [2, 3, 4] で再実装したライブラリです。

## 概要

`jingsongliujing/OnnxOCR` [1] は、BaiduのPaddleOCR [5] をベースにした高性能な軽量OCRソリューションです。しかし、標準的なRustでの利用（例: `ort` [6, 7] クレート）は、MicrosoftのC++製 `onnxruntime` [2] へのFFI（Foreign Function Interface）バインディングに依存します。

**`pure-onnx-ocr` は、このC++依存を完全に排除します。**

  * **Pure Rust**: C/C++のビルドツールチェインや共有ライブラリ（`.dll`, `.so`, `.dylib`）は一切不要です。`cargo build` だけで動作します。
  * **推論エンジン**: `onnxruntime` [2] の代わりに、Pure Rustの高速な推論エンジン `tract` [8, 9, 10] を使用します。
  * **画像処理**: `opencv-python` [11] の依存を排除し、Pure Rustの `image` [12, 13, 14] クレートと `imageproc` [15, 16, 17, 18, 19] クレートを使用します。
  * **ジオメトリ処理**: `pyclipper` [20, 21, 22, 23, 24] (C++ラッパー) や `shapely` [20, 25, 26] (GEOS C APIラッパー) の依存を排除し、Pure Rustの `i_overlay` [27, 28, 29, 30] と `geo-types` [31, 32, 33, 34, 35] を使用します。

これにより、組み込み、WASM、サーバーレス環境など、C++依存が障壁となるシナリオでも容易にOCR機能をデプロイできます。

## インストール方法

`Cargo.toml` に以下の行を追加してください。

```toml
[dependencies]
pure_onnx_ocr = "0.1.0" # (最新のバージョンを指定してください)

# OCRの結果（特に座標）を扱うために、以下のクレートも推奨されます
image = "0.25"
geo-types = "0.7"
```

## クイックスタート

### 1\. 必要なモデルの準備

本ライブラリは推論エンジンのみを提供します。モデルファイルは別途準備する必要があります。
`jingsongliujing/OnnxOCR` [1] が推奨する `PP-OCRv5_Server-ONNX` [36, 20, 37] モデルセット（またはMobile版 [1]）をダウンロードしてください。

以下の3つのファイルが必要です。

  * `det.onnx`: テキスト検出モデル (DBNet) [38]
  * `rec.onnx`: テキスト認識モデル (SVTR\_HGNet) [39, 37]
  * `ppocrv5_dict.txt`: 文字辞書ファイル [40, 41, 42]

### 2\. サンプルコード

```rust
use pure_onnx_ocr::{OcrEngine, OcrEngineBuilder, OcrError, OcrResult, Polygon};
use std::path::Path;

fn main() -> Result<(), OcrError> {
    // 1. ビルダーを使用してエンジンを構築します (アプリケーション起動時に1回だけ)
    //
    // 警告: `rec.onnx` (SVTRモデル) のロードは、`tract` が
    // `LayerNormalization` [43, 44, 45] や `Scan` [46] などの
    // 複雑なオペレータをサポートしているかに依存します。
    // `build()` が失敗する場合、互換性のないオペレータが原因である可能性があります。
    let engine = OcrEngineBuilder::new()
      .det_model_path("models/ppocrv5/det.onnx")
      .rec_model_path("models/ppocrv5/rec.onnx")
      .dictionary_path("models/ppocrv5/ppocrv5_dict.txt") [40, 41]
       // (オプション) パラメータをカスタマイズ
      .det_limit_side_len(960) // [47, 48]
      .det_unclip_ratio(1.5)   // [20]
      .rec_batch_size(8)
      .build()
      .expect("OCRエンジンのビルドに失敗しました。モデルパスや辞書パスを確認してください。");

    println!("Pure Rust OCR エンジンが正常にロードされました。");

    // 2. 画像ファイルパスを指定してOCRを実行します
    let image_path = "path/to/your/image.jpg";
    let results: Vec<OcrResult> = engine.run_from_path(image_path)?;

    println!("画像 '{}' から {} 個のテキスト領域を検出しました。", image_path, results.len());

    // 3. 結果を処理します
    for result in results {
        println!("---------------------------------");
        println!("  テキスト: {}", result.text);
        println!("  信頼度: {:.4}", result.confidence);
        // `bounding_box` は `geo_types::Polygon` です
        println!("  座標 (外周): {:?}", result.bounding_box.exterior().points());
    }

    Ok(())
}
```

## 開発進捗

- 2025-11-09: `tract-onnx` を用いた `det.onnx` (DBNet) のロードとダミー推論 PoC (`task-poc-001`) を完了しました。`models/ppocrv5/det.onnx` を読み込み、ゼロ埋め入力テンソルで 1 回推論が通ることを確認しています。
- 2025-11-09: `rec.onnx` (SVTR\_HGNet) のロードとダミー推論 PoC (`task-poc-002`) を実施しました。`[1, 3, 48, 320]` の擬似画像テンソルで推論し、出力形状 `[1, 40, 18385]` と値域 `0.0..0.981465` を確認しています。`tract` の `into_decluttered` 最適化に約6分を要するためパフォーマンス改善が今後の課題です。
- 2025-11-09: 検出前処理モジュール `DetPreProcessor` (`task-det-001`) を実装。長辺制限リサイズ、正規化、NCHW変換に対応し、代表的なケースをカバーするユニットテストを追加しました。
- 2025-11-09: DBNet 推論モジュール `DetInferenceSession` (`task-det-002`) を実装し、入力解像度ごとに最適化済みランナブルをキャッシュする仕組みと確率マップ抽出ロジックを追加しました。
- 2025-11-09: 検出後処理（輪郭抽出）モジュール `DetPostProcessor` (`task-det-003`) を実装し、閾値処理と面積フィルタリングによる輪郭抽出を追加しました。
- 2025-11-09: 検出後処理（ポリゴン拡張）モジュール `DetPolygonUnclipper` (`task-det-004`) を実装し、`i_overlay` によるバッファリングと面積フィルタで DBNet の Unclip 処理を再現しました。
- 2025-11-09: 検出後処理（座標復元）モジュール `DetPolygonScaler` (`task-det-005`) を実装し、リサイズ比に基づく逆スケーリングと丸め・クリッピングを追加しました。
- 2025-11-09: 認識前処理モジュール `RecPreProcessor` (`task-rec-001`) を実装し、矩形クロップと強制リサイズ、正規化、NCHWバッチ化、パディング処理、およびユニットテストを追加しました。
- 2025-11-09: 認識推論モジュール `RecInferenceSession` (`task-rec-002`) を実装し、`tract-onnx` によるバッチ推論、入力形状に応じたランナブルキャッシュ、出力ロジットの `ndarray` 変換、および有効タイムステップ計算を追加しました。
- 2025-11-09: 認識辞書ローダー `RecDictionary` (`task-rec-003`) を実装し、UTF-8 辞書ファイルのロード、重複検知、インデックス/トークン相互参照、およびユニットテストを整備しました。
- 2025-11-09: CTC Greedy デコーダー `CtcGreedyDecoder` (`task-rec-004`) を実装し、重複圧縮とブランク除去、確率に基づく信頼度計算、辞書連携、およびユニットテストを追加しました。

## APIリファレンス (要約)

### `OcrEngineBuilder`

`OcrEngine` を安全に構築するためのビルダー。

  * `OcrEngineBuilder::new() -> Self`
  * `det_model_path(...) -> Self`: 検出モデル (`det.onnx`) のパスを設定
  * `rec_model_path(...) -> Self`: 認識モデル (`rec.onnx`) のパスを設定
  * `dictionary_path(...) -> Self`: 辞書ファイル (`.txt`) のパスを設定
  * `det_limit_side_len(u32) -> Self`: (オプション) 検出時の画像最大辺長 [47, 48]
  * `det_unclip_ratio(f64) -> Self`: (オプション) 検出領域の拡大率 [20]
  * `rec_batch_size(usize) -> Self`: (オプション) 認識時のバッチサイズ
  * `build() -> Result<OcrEngine, OcrError>`: モデルをロードしてエンジンを構築

### `OcrEngine`

スレッドセーフなOCR実行エンジン。`OcrEngineBuilder` を介してのみ作成可能です。

  * `run_from_path<P: AsRef<Path>>(&self, path: P) -> Result<Vec<OcrResult>, OcrError>`:
    ファイルパスから画像をロードしてOCRを実行します。
  * `run_from_image(&self, image: &image::DynamicImage) -> Result<Vec<OcrResult>, OcrError>`:
    メモリ上の `image::DynamicImage` [49] からOCRを実行します。

### `OcrResult` (Struct)

単一のOCR結果を保持する構造体 [50]。

  * `pub text: String`: 認識されたテキスト
  * `pub confidence: f32`: 信頼度 (0.0 \~ 1.0)
  * `pub bounding_box: Polygon`: 座標 (`geo_types::Polygon` [31, 34] 型)

### `OcrError` (Enum)

ライブラリ内で発生する可能性のあるエラー。

  * `IoError(...)`: ファイルI/Oエラー
  * `ImageError(...)`: `image` クレート [12, 13, 14] によるデコードエラー
  * `ModelLoadError(String)`: `tract` [9, 10] によるモデルロード失敗 (オペレータ非互換 [43, 46] など)
  * `InferenceError(String)`: 推論実行時のエラー
  * `ProcessingError(String)`: 前後処理ロジックのエラー

## 貢献 (Contributing)

バグ報告、機能リクエスト、プルリクエストを歓迎します。
IssueやPRを作成する前に、既存のものを確認してください。

## ライセンス

このプロジェクトは、以下のいずれかのライセンスの下で利用可能です。

  * **Apache License, Version 2.0** ((LICENSE-APACHE) または [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))

これはリファレンス元である `PaddleOCR` [20], `OnnxOCR` [1], `tract` [10] のライセンス（主にApache-2.0）に基づいています。