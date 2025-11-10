---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M2, dictionary, io]
depends_on:
---

# タスク概要
`ppocrv5_dict.txt` をロードして文字辞書を構築する機能を実装する。

## 要件
- UTF-8 辞書ファイルを安全に読み込み、改行や空行を適切に処理する
- インデックスと文字のマッピングを保持するデータ構造を定義する
- 異常な入力に対するバリデーションとエラーメッセージを実装する
- 単体テストで辞書サイズとサンプル取得を検証する

## 作業ログ
- 2025-11-09: `RecDictionary` と `DictionaryError` を `src/dictionary.rs` に実装し、UTF-8 辞書ファイルのロード、トリミング、空行スキップ、重複検知、インデックス/トークン検索を追加。
- 2025-11-09: 辞書ロードのユニットテストを実装し、正常系・空ファイル・重複エントリのエラーケースをカバー。

## テスト
- 2025-11-09: `cargo test`

