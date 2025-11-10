---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M1, preprocessing, dbnet]
depends_on:
---

# タスク概要
DBNet 検出パイプラインの前処理 `DetPreProcessor` を実装し、入力画像をモデル互換なテンソルへ変換する。

## 要件
- 設計仕様に基づきリサイズ、正規化、チャネル順変換 (NHWC -> NCHW) を実装する
- 境界ケースを含むユニットテストを追加する
- 出力テンソル形状と dtype がモデル要件と一致することを確認する
- 実装内容と設計上の判断を開発ログに記録する

## 作業ログ
- 2025-11-09: `DetPreProcessor` を `src/preprocessing.rs` に実装。DBNet 入力に合わせて長辺制限リサイズ、正規化、NCHW 変換を行うロジックを追加。
- 2025-11-09: 代表的な画像サイズ（長辺制限、未リサイズ、正規化確認）をカバーするユニットテストを追加。

## テスト
- 2025-11-09: `cargo test`

