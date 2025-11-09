---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M2, inference, tract]
depends_on:
---

# タスク概要
`tract-onnx` を利用し SVTR 認識モデルのバッチ推論を行うコンポーネントを実装する。

## 要件
- モデルセッションの初期化とキャッシュ戦略を設計する
- 可変長バッチ入力を適切にトリミングして推論する
- 出力テンソルを CTC デコードで扱いやすい `ndarray` 形式へ変換する
- 失敗時のリカバリとログを整備する

## 作業ログ
- 2025-11-09: `RecInferenceSession` を `src/recognition.rs` に実装し、入力形状に基づくランナブルキャッシュ、`tract-onnx` 実行、出力ロジットの `Array3<f32>` 化、および有効タイムステップ算出を追加。
- 2025-11-09: 認識推論の単体テストを追加し、実モデル `models/ppocrv5/rec.onnx` を用いたバッチ推論が正しく動作することを確認。

## テスト
- 2025-11-09: `cargo test`

