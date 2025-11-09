# インターフェース設計書 (API設計書): Pure Rust OnnxOCR

作成者: Shion Watanabe  
日付: 2025-11-09  
リポジトリ: http://github.com/siska-tech/pure-onnx-ocr

## 🎯 目的

  * **プログラム利用者が直接呼び出す、公開API（関数、クラス、メソッド）を厳密に定義する。**
    本ライブラリ（クレート）の公開インターフェースは、`OcrEngineBuilder`（設定用）、`OcrEngine`（実行用）、`OcrResult`（結果用）、`OcrError`（エラー用）に集約されます。

  * **利用者が直感的かつ安全にプログラムを使えるようにする。**
    アーキテクチャ設計で定義した「Facadeパターン」を採用します。`OcrEngineBuilder` を用いて、モデルのロードや設定といった複雑な初期化プロセスを安全に実行します。初期化が完了した `OcrEngine` は、`run_from_path` や `run_from_image` といったシンプルなメソッドを提供するだけで、利用者は内部の2段階のOCRパイプライン（検出・認識）を意識する必要はありません。

-----

## 記載すべき項目

### 1\. 公開API一覧

#### 名前空間 / モジュール

`pure_onnx_ocr`

#### クラス（Struct）と公開メソッド

  * **`OcrEngine`**
      * `pub fn run_from_path<P: AsRef<Path>>(&self, path: P) -> Result<Vec<OcrResult>, OcrError>`
      * `pub fn run_from_image(&self, image: &image::DynamicImage) -> Result<Vec<OcrResult>, OcrError>`
  * **`OcrEngineBuilder`**
      * `pub fn new() -> Self`
      * `pub fn det_model_path<P: AsRef<Path>>(self, path: P) -> Self`
      * `pub fn rec_model_path<P: AsRef<Path>>(self, path: P) -> Self`
      * `pub fn dictionary_path<P: AsRef<Path>>(self, path: P) -> Self`
      * `pub fn det_limit_side_len(self, len: u32) -> Self`
      * `pub fn det_unclip_ratio(self, ratio: f64) -> Self`
      * `pub fn rec_batch_size(self, size: usize) -> Self`
      * `pub fn build(self) -> Result<OcrEngine, OcrError>`
  * **`OcrResult`**
      * `pub text: String`
      * `pub confidence: f32`
      * `pub bounding_box: Polygon`

#### 公開する列挙型（Enum）

  * **`OcrError`**
      * `IoError(std::io::Error)`
      * `ImageError(image::ImageError)`
      * `ModelLoadError(String)`
      * `InferenceError(String)`
      * `ProcessingError(String)`

#### 公開する定数

(なし)

#### 再エクスポート (Re-exports)

利用者の利便性のため、`geo-types` の主要な型をクレートのトップレベルで再エクスポートします。

  * `pub use geo_types::{Point, Polygon};`

-----

### 2\. 各APIの詳細定義

#### `OcrError` (Enum)

  * **概要**:
    ライブラリ内で発生する可能性のあるすべてのエラーをカプセル化した公開エラー型。`std::error::Error` を実装します。
  * **バリアント**:
      * `IoError(std::io::Error)`: モデルファイルや辞書ファイル [2, 3, 4]、画像ファイルの読み込みに失敗した場合。
      * `ImageError(image::ImageError)`: `image` クレート [5] が画像のデコードに失敗した場合。
      * `ModelLoadError(String)`: `tract-onnx` [6] がONNXモデルファイル（`det.onnx` または `rec.onnx`）の解析またはロードに失敗した場合（例：オペレータ非互換 [7, 8]）。
      * `InferenceError(String)`: `tract-onnx` がモデルの実行（推論）に失敗した場合。
      * `ProcessingError(String)`: 前処理または後処理のロジック（輪郭抽出 [9, 10]、CTCデコード [11] など）で予期せぬエラーが発生した場合。

-----

#### `OcrResult` (Struct)

  * **概要**:
    画像から検出・認識された単一のテキスト領域の結果を格納する構造体 [12]。
  * **フィールド**:
      * `pub text: String`
          * 認識されたUTF-8文字列。
      * `pub confidence: f32`
          * 認識の信頼度（認識モデルが信頼度を出力しない場合、`0.0` または `1.0` が入る可能性があります）。
      * `pub bounding_box: Polygon`
          * テキスト領域を囲む多角形の座標 [13, 14]。`pure_onnx_ocr::Polygon` 型（`geo-types::Polygon` の再エクスポート）です。

-----

#### `OcrEngineBuilder` (Struct)

  * **概要**:
    `OcrEngine` を安全に構築するためのビルダー。モデルパスや各種パラメータを設定します。

  * **`pub fn new() -> Self`**

      * **シグネチャ**: `pub fn new() -> Self`
      * **概要**: ビルダーの新しいインスタンスをデフォルト値で作成します。
      * **戻り値**: `OcrEngineBuilder`

  * **`pub fn det_model_path<P: AsRef<Path>>(self, path: P) -> Self`**

      * **シグネチャ**: `pub fn det_model_path<P: AsRef<Path>>(self, path: P) -> Self`
      * **概要**: テキスト検出（DBNet）[15] の `.onnx` モデルファイルへのパスを設定します。
      * **引数**: `path`: `.onnx` ファイルへのパス。

  * **`pub fn rec_model_path<P: AsRef<Path>>(self, path: P) -> Self`**

      * **シグネチャ**: `pub fn rec_model_path<P: AsRef<Path>>(self, path: P) -> Self`
      * **概要**: テキスト認識（SVTR）[2] の `.onnx` モデルファイルへのパスを設定します。
      * **引数**: `path`: `.onnx` ファイルへのパス。

  * **`pub fn dictionary_path<P: AsRef<Path>>(self, path: P) -> Self`**

      * **シグネチャ**: `pub fn dictionary_path<P: AsRef<Path>>(self, path: P) -> Self`
      * **概要**: CTCデコード [11] に使用する文字辞書ファイル（例: `ppocrv5_dict.txt` [2, 3, 4]）へのパスを設定します。
      * **引数**: `path`: 辞書テキストファイルへのパス。

  * **`pub fn det_limit_side_len(self, len: u32) -> Self`**

      * **シグネチャ**: `pub fn det_limit_side_len(self, len: u32) -> Self`
      * **概要**: 検出前処理で、画像の長辺をリサイズする最大長さを設定します [16, 17]。
      * **引数**: `len`: 最大ピクセル長。（デフォルト: `960`）

  * **`pub fn det_unclip_ratio(self, ratio: f64) -> Self`**

      * **シグネチャ**: `pub fn det_unclip_ratio(self, ratio: f64) -> Self`
      * **概要**: 検出後処理で、バウンディングボックスを拡大する比率（unclip ratio）[18] を設定します。
      * **引数**: `ratio`: 拡大比率。（デフォルト: `1.5`）

  * **`pub fn rec_batch_size(self, size: usize) -> Self`**

      * **シグネチャ**: `pub fn rec_batch_size(self, size: usize) -> Self`
      * **概要**: 認識モデルを実行する際の最大バッチサイズを設定します。
      * **引数**: `size`: バッチサイズ。（デフォルト: `8`）

  * **`pub fn build(self) -> Result<OcrEngine, OcrError>`**

      * **シグネチャ**: `pub fn build(self) -> Result<OcrEngine, OcrError>`
      * **概要**:
        設定されたパスから全てのモデルと辞書をロードし、推論の準備が整った `OcrEngine` インスタンスを生成します。
        このプロセスで `tract-onnx` [6] によるモデルのロードと最適化が実行されます。
      * **戻り値**: 成功した場合は `OcrEngine`、失敗した場合は `OcrError` を返します。
      * **エラー**:
          * `OcrError::IoError`: `det_model_path`、`rec_model_path`、`dictionary_path` [2, 3] のいずれかのファイルが見つからないか、読み取り権限がない場合。
          * `OcrError::ModelLoadError`: ONNXモデルが不正であるか、`tract` がサポートしないオペレータ [7, 8] を含んでいる場合。

-----

#### `OcrEngine` (Struct)

  * **概要**:
    ロード済みのモデルと設定を保持する、スレッドセーフなOCR実行エンジン。この構造体のフィールドはすべてプライベートであり、`OcrEngineBuilder` を介してのみ作成可能です。

  * **`pub fn run_from_path<P: AsRef<Path>>(&self, path: P) -> Result<Vec<OcrResult>, OcrError>`**

      * **シグネチャ**: `pub fn run_from_path<P: AsRef<Path>>(&self, path: P) -> Result<Vec<OcrResult>, OcrError>`
      * **概要**: ファイルパスから画像をロードし、完全なOCRパイプライン（検出＋認識）を実行します。
      * **引数**: `path`: 認識対象の画像ファイル（JPEG, PNG等）へのパス。
      * **戻り値**: 検出されたすべてのテキスト領域の結果 `Vec<OcrResult>`。テキストが検出されなかった場合は空のVec。
      * **エラー**:
          * `OcrError::IoError`: `path` が存在しない場合。
          * `OcrError::ImageError`: ファイルが有効な画像でない場合 [5]。
          * `OcrError::InferenceError`: 推論実行中に `tract` がエラーを返した場合。

  * **`pub fn run_from_image(&self, image: &image::DynamicImage) -> Result<Vec<OcrResult>, OcrError>`**

      * **シグネチャ**: `pub fn run_from_image(&self, image: &image::DynamicImage) -> Result<Vec<OcrResult>, OcrError>`
      * **概要**: 既にメモリ上にある `image::DynamicImage` [5] に対して、完全なOCRパイプラインを実行します。
      * **引数**: `image`: `image` クレートの `DynamicImage` 型の画像バッファへの参照。
      * **戻り値**: 検出されたすべてのテキスト領域の結果 `Vec<OcrResult>`。
      * **エラー**:
          * `OcrError::InferenceError`: 推論実行中に `tract` がエラーを返した場合。

-----

### 3\. 使用例（Code Snippet）

#### `Cargo.toml`

```toml
[dependencies]
pure_onnx_ocr = "0.1.0" # (このクレートの想定名)
image = "0.25"
geo-types = "0.7" # OcrResult の Polygon を操作する場合に必要
```

#### `src/main.rs`

```rust
use pure_onnx_ocr::{OcrEngine, OcrEngineBuilder, OcrError, OcrResult, Polygon}; // Polygon は再エクスポートされる
use std::path::Path;

fn main() -> Result<(), OcrError> {
    // 1. ビルダーを使用してエンジンを構築する
    // このステップは一度だけ（例：アプリケーション起動時）実行します。
    // モデルと辞書のパスは、jingsongliujing/OnnxOCR [1] からダウンロードした
    // PP-OCRv5_Server-ONNX [2] モデルのパスを指していると仮定します。
    let engine = OcrEngineBuilder::new()
       .det_model_path("models/ppocrv5/det.onnx")
       .rec_model_path("models/ppocrv5/rec.onnx")
       .dictionary_path("models/ppocrv5/ppocrv5_dict.txt") [2, 3]
       .det_limit_side_len(960) // オプション: デフォルト値 [16]
       .det_unclip_ratio(1.5)   // オプション: デフォルト値 [18]
       .rec_batch_size(8)       // オプション: デフォルト値
       .build()
       .expect("OCRエンジンのビルドに失敗しました。モデルパスや辞書パスを確認してください。");

    println!("Pure Rust OCR エンジンが正常にロードされました。");

    // 2. 画像ファイルパスを指定してOCRを実行する
    let image_path = "path/to/your/image.jpg";
    let results: Vec<OcrResult> = engine.run_from_path(image_path)?;

    println!("画像 '{}' から {} 個のテキスト領域を検出しました。", image_path, results.len());

    // 3. 結果を処理する
    for result in results {
        println!("---------------------------------");
        println!("  テキスト: {}", result.text);
        println!("  信頼度: {:.4}", result.confidence);
        // OcrResult.bounding_box は geo_types::Polygon [13, 14]
        println!("  座標 (外周): {:?}", result.bounding_box.exterior().points());
    }

    // ----------------------------------------------------
    // (オプション) メモリ上の画像から直接実行する
    // ----------------------------------------------------
    /*
    let image_bytes = std::fs::read(image_path)?;
    let dynamic_image = image::load_from_memory(&image_bytes)
       .map_err(|e| OcrError::ImageError(e))?;
    
    let results_from_memory: Vec<OcrResult> = engine.run_from_image(&dynamic_image)?;
    
    println!("\nメモリ上の画像から {} 個のテキスト領域を検出しました。", results_from_memory.len());
    for result in results_from_memory {
         println!("  テキスト: {}", result.text);
    }
    */

    Ok(())
}
```