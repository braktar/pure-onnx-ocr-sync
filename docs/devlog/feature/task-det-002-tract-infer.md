---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M1, inference, tract]
depends_on:
---

# タスク概要
`tract-onnx` を利用して DBNet モデルの検出推論を実行するモジュールを実装する。

## 要件
- 既存の前処理出力を `tract-onnx` に入力できるよう橋渡しコードを作成する
- モデルの初期化とセッション再利用の仕組みを整備する
- 推論結果を後処理で扱いやすい配列形式に変換する
- 主要なエラーハンドリングとログ出力を追加する

## 作業ログ
- 2025-11-09: DBNet 推論用セッション `DetInferenceSession` を `src/detection.rs` に実装。`InferenceModel` を保持し、入力解像度ごとに最適化プランをキャッシュする仕組みを導入。
- 2025-11-09: 前処理出力から `Array2<f32>` の確率マップへ変換するラッパーを追加し、エラーハンドリングとログ出力を整備。
- 2025-11-09: ONNX モデル存在を確認するユニットテスト `detection_inference_runs` を追加し、動的リサイズを含む推論パスを検証。

## テスト
- 2025-11-09: `cargo test detection::tests::detection_inference_runs -- --test-threads=1`
- 2025-11-09: `cargo test`

