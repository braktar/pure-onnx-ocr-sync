# `pure-onnx-ocr` (Pure Rust OnnxOCR)

作成者: Shion Watanabe  
日付: 2025-11-09  
リポジトリ: http://github.com/siska-tech/pure-onnx-ocr

Pure RustでOCRパイプラインを構築するためのライブラリです。Baidu PaddleOCR由来の `det.onnx` (DBNet) と `rec.onnx` (SVTR_HGNet) を Pure Rust エコシステムのみで実行できるよう再設計しています。

> **English documentation is available in `README_en.md`.**  
> Other architectural documents also provide English counterparts (see [Documentation](#documentation)).

## 特長

- **Pure Rustのみで完結**: C/C++製オンプレミスライブラリやFFIの導入が不要です。`cargo build` だけでセットアップできます。
- **DBNet + SVTR パイプライン**: PaddleOCRが採用する検出・認識モデルをRust上で再現します。
- **モジュール構成が明確**: `OcrEngineBuilder` と `OcrEngine` を中心に、前処理・推論・後処理を分離しています。
- **移植性**: 組み込み環境、サーバーレス、WASMなど、C++依存が課題となる環境でも動作を想定しています。

## 導入手順

### 1. 前提条件

- Rust 1.75 以降 (stable)
- CPU推論を想定した x86\_64 / aarch64 環境
- ONNXモデルファイル (`det.onnx`, `rec.onnx`) と辞書ファイル (`ppocrv5_dict.txt`)

### 2. 依存関係の追加

`Cargo.toml` の `[dependencies]` に以下を追加してください。

```toml
[dependencies]
pure_onnx_ocr = "0.1.0"         # crates.io リリース後に最新バージョンへ更新してください
image = "0.25"                  # OCR結果の描画や前処理に利用する場合
geo-types = "0.7"               # ポリゴン座標の操作に利用する場合
```

### 3. モデルと辞書の配置

1. PaddleOCR配布の `PP-OCRv5_Server-ONNX` もしくは `PP-OCRv5_Mobile-ONNX` をダウンロードします。
2. 本プロジェクトの `models/` ディレクトリなど、任意の場所に以下のファイルを配置してください。
   - `models/ppocrv5/det.onnx`
   - `models/ppocrv5/rec.onnx`
   - `models/ppocrv5/ppocrv5_dict.txt`
3. `OcrEngineBuilder` へ上記パスを渡すことで推論が可能になります。

## クイックスタート

```rust
use pure_onnx_ocr::{OcrEngineBuilder, OcrError, OcrResult};

fn main() -> Result<(), OcrError> {
    // 1. エンジンの初期化（アプリケーション起動時に一度だけ実行）
    let engine = OcrEngineBuilder::new()
        .det_model_path("models/ppocrv5/det.onnx")
        .rec_model_path("models/ppocrv5/rec.onnx")
        .dictionary_path("models/ppocrv5/ppocrv5_dict.txt")
        .det_limit_side_len(960)   // 任意調整: 入力画像の最大長辺
        .det_unclip_ratio(1.5)     // 任意調整: 検出ポリゴンのオフセット率
        .rec_batch_size(8)         // 任意調整: 認識推論のバッチサイズ
        .build()?;

    // 2. 画像ファイルからOCRを実行
    let results: Vec<OcrResult> = engine.run_from_path("examples/demo.jpg")?;

    // 3. OCR結果を活用
    for (idx, result) in results.iter().enumerate() {
        println!("#{} text={} confidence={:.4}", idx, result.text, result.confidence);
        println!("   polygon={:?}", result.bounding_box.exterior().points());
    }

    Ok(())
}
```

### よくあるエラー

- `ModelLoad`: `tract` が未対応のONNXオペレータ（例: `LayerNormalization`, `Scan`）を検出した場合に発生します。
- `Dictionary`: 辞書ファイルの文字コードがUTF-8以外の場合に発生します。UTF-8 (BOM無し) で保存してください。

## API概要

`pure_onnx_ocr` クレートは次の構造体・列挙型を公開します。

| シンボル           | 概要                                                                                           |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| `OcrEngineBuilder` | モデル・辞書・パラメータを設定し、`OcrEngine` を構築するためのビルダー。                       |
| `OcrEngine`        | 検出・認識パイプラインを統合したファサード。`run_from_path` と `run_from_image` を提供します。 |
| `OcrResult`        | 認識された単一テキスト領域の結果 (`text`, `confidence`, `bounding_box`) を保持します。         |
| `OcrError`         | ライブラリ全体で発生し得るエラーをカプセル化した列挙型です。                                   |
| `Polygon`          | `geo-types::Polygon` の再エクスポート。検出結果の座標表現に利用します。                        |

詳細なAPI仕様については `docs/interface_design.md` および `docs/interface_design_en.md` を参照してください。

## Documentation

- `docs/architecture.md` / `docs/architecture_en.md`
- `docs/detail_design.md` / `docs/detail_design_en.md`
- `docs/interface_design.md` / `docs/interface_design_en.md`
- `docs/requirements.md` / `docs/requirements_en.md`
- `docs/references.md` / `docs/references_en.md`
- `docs/test_specification.md` / `docs/test_specification_en.md`

ドキュメントセット全体の英語版を整備し、国際的なコントリビューターでも参照可能な構成としています。

## 開発進捗

- 2025-11-09: `tract-onnx` を用いた `det.onnx` (DBNet) のロードとダミー推論 PoC (`task-poc-001`) を完了。
- 2025-11-09: `rec.onnx` (SVTR_HGNet) のロードとダミー推論 PoC (`task-poc-002`) を完了。出力形状 `[1, 40, 18385]` を確認。
- 2025-11-09: 検出前処理 `DetPreProcessor` (`task-det-001`) を実装。長辺制限リサイズ、正規化、NCHW変換に対応。
- 2025-11-09: DBNet 推論モジュール `DetInferenceSession` (`task-det-002`) を実装。解像度別にランナブルをキャッシュ。
- 2025-11-09: 検出後処理 `DetPostProcessor` (`task-det-003`) を実装。閾値処理と輪郭抽出を追加。
- 2025-11-09: ポリゴン拡張 `DetPolygonUnclipper` (`task-det-004`) を実装。`i_overlay` を利用したバッファリングを実現。
- 2025-11-09: 座標復元 `DetPolygonScaler` (`task-det-005`) を実装。逆スケーリングと丸め処理を追加。
- 2025-11-09: 認識前処理 `RecPreProcessor` (`task-rec-001`) を実装。クロップ、強制リサイズ、バッチ化を統合。
- 2025-11-09: 認識推論 `RecInferenceSession` (`task-rec-002`) を実装。`tract-onnx` によるバッチ推論を整備。
- 2025-11-09: 辞書ローダー `RecDictionary` (`task-rec-003`) を実装。重複検知やマッピングを追加。
- 2025-11-09: CTC Greedy デコーダー `CtcGreedyDecoder` (`task-rec-004`) を実装。重複圧縮とブランク除去をサポート。
- 2025-11-09: 認識ポストプロセッサ `RecPostProcessor` (`task-rec-005`) を実装。ロジット処理と辞書マッピングを統合。
- 2025-11-09: 公開ビルダー `OcrEngineBuilder` (`task-api-001`) を実装。パラメータ検証とモデル初期化を実装。
- 2025-11-09: `OcrEngine` ファサード (`task-api-002`) を実装。検出・認識パイプラインを統合。
- 2025-11-09: `OcrEngine::run_from_path` (`task-api-003`) を実装し、E2E OCR処理を完成。
- 2025-11-09: `OcrEngine::run_from_image` (`task-api-004`) を実装し、メモリ上の画像入力に対応。
- 2025-11-09: 公開エラー型 `OcrError` (`task-api-005`) を整備し、エラーパスを統一。
- 2025-11-09: ドキュメント整備タスク `task-doc-001` を完了。READMEの再構成と英語版ドキュメントを追加。
- 2025-11-09: 公開APIの Rustdoc コメント (`task-doc-002`) を整備し、`cargo doc` で生成物を確認。

## コントリビューション

Pull Request や Issue を歓迎します。大規模な変更を提案する場合は、まず Issue で背景と目的を共有してください。

### 開発フローの指針

- `cargo fmt` と `cargo clippy` でスタイルと静的解析を行ってから PR を作成してください。
- 追加した機能には可能な限りユニットテストを付与してください。
- ドキュメント更新の場合は `docs/devlog/ROADMAP.md` と関連タスクの進捗を同期してください。

## ライセンス

本プロジェクトは `Apache-2.0` ライセンスで提供します。リファレンス実装である `PaddleOCR`, `OnnxOCR`, `tract` と同一ファミリーのライセンス体系に準拠します。