---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M2, postprocessing, decode]
depends_on:
---

# タスク概要
`ndarray::argmax` の結果を CTC Greedy デコードと結合し、最終的なテキスト文字列のリストを返す処理を実装する。

## 要件
- バッチごとのトークン列を辞書文字にマッピングする
- デコード処理と辞書の同期を保証するエラーハンドリングを実装する
- 部分的な認識失敗時のフォールバックポリシーを定義する
- 単体テストで複数バッチケースを検証する

## 作業ログ
- 2025-11-09: `RecPostProcessor` を `src/recognition.rs` に追加し、`RecInferenceOutput` から `CtcGreedyDecoder` を介してテキスト列を生成する統合処理を実装。
- 2025-11-09: `CtcGreedyDecoderConfig` にフォールバックトークン設定を導入し、辞書外インデックス発生時の `[UNK]` 代替出力とカウント追跡を追加。
- 2025-11-09: 認識ポストプロセッサ単体テストを実装し、複数バッチとフォールバック、重複抑制を検証。

## テスト
- 2025-11-09: `cargo test`

