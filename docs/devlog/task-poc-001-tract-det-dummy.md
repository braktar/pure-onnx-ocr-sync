---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M0, tract, dbnet]
depends_on:
---

# タスク概要
`tract-onnx` の推論エンジンを使って `det.onnx` (DBNet) モデルを Pure Rust 環境でロードし、単純なダミー入力で推論を実行できることを確認する。

## 要件
- `tract-onnx` を利用した DBNet モデルのロード処理を実装する
- モデル入力テンソルのシェイプを満たすダミー入力データを生成する
- 推論を 1 回実行しクラッシュせずに出力が得られることを確認する
- 実装手順と結果を開発ログに記録する

## 実装メモ
- `cargo init --lib` でライブラリクレートを初期化し、`tract-onnx` と `ndarray` を依存関係に追加した。
- `run_dbnet_dummy_inference` を実装し、`det.onnx` をロードしてダミー入力 (`1x3x320x320`) で 1 回推論を実行する PoC を構築。
- `InferenceFact::from(&dummy_input)` を用いて入力シェイプを `tract` に伝え、`into_typed`→`into_decluttered`→`into_optimized`→`into_runnable` のパイプラインでモデルを実行可能にした。
- ユニットテスト `dbnet_dummy_inference_runs_successfully` を追加し、実際のモデルファイルを使って PoC がクラッシュせず出力を返すことを確認。
- テストは `cargo test` で実行し、推論完了まで約 67 秒かかったが正常終了した。

