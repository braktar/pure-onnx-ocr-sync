---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M2, preprocessing, rec]
depends_on:
---

# タスク概要
`RecPreProcessor` を実装し、クロップ画像の整形・正規化・バッチング機構を提供する。

## 要件
- 入力矩形に基づくクロップ処理を実装する
- モデル仕様に沿って強制リサイズとアスペクト比処理を行う
- Batched NCHW テンソルを生成し複数サンプル間のパディングを管理する
- 単体テストで単一・複数テキストラインをカバーする

## 作業ログ
- 2025-11-09: `RecPreProcessor`、設定オプション、エラー型、`PreprocessedRecBatch` を `src/preprocessing.rs` に追加し、矩形クロップ、強制リサイズ、チャンネルごとの正規化、パディング付きNCHWバッチ生成を実装。
- 2025-11-09: 単一・複数領域、およびエラーケースをカバーするユニットテストを追加し、バッチ幅とパディングが仕様どおりに動くことを検証。

## テスト
- 2025-11-09: `cargo test`

