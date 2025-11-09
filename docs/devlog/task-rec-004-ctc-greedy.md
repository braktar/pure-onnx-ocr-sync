---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M2, postprocessing, ctc]
depends_on:
---

# タスク概要
Pure Rust で CTC Greedy デコードアルゴリズムを実装し、SVTR 出力を文字 ID 列へ変換する。

## 要件
- Time-major ログ確率配列から最大値インデックスを抽出するロジックを実装する
- 重複文字の圧縮と空白トークンの削除ルールを整理する
- 典型的な入力ケースと境界ケースの単体テストを作成する
- ベンチマークの指標を収集し性能リスクを明示する

## 作業ログ
- 2025-11-09: `CtcGreedyDecoder` と `DecodedSequence` を `src/ctc.rs` に実装し、ブランクトークン検出、重複抑制、確率ベースの信頼度算出、辞書連携を追加。
- 2025-11-09: ユニットテストで重複圧縮、ブランクのみ入力、ブランクID範囲外エラーなどのケースを検証。

## テスト
- 2025-11-09: `cargo test`

