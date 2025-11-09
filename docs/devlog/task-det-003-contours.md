---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M1, postprocessing, contours]
depends_on:
---

# タスク概要
DBNet の確信度マップから `imageproc::find_contours` を用いてテキスト候補領域の輪郭を抽出する。

## 要件
- 推論出力からバイナリマスクを生成するしきい値処理を実装する
- `imageproc::contours` API の選定とラップ処理を行う
- ノイズ除去や最小面積閾値を含めたフィルタリングロジックを追加する
- 主要ケースをカバーするユニットテストを作成する

## 作業ログ
- 2025-11-09: `Cargo.toml` に `imageproc = "0.25"` を追加して輪郭抽出の依存関係を導入。
- 2025-11-09: `src/postprocessing.rs` に `DetPostProcessor` を実装し、確率マップから二値マスク生成・輪郭抽出・面積フィルタリングを行うロジックを追加。
- 2025-11-09: 輪郭抽出結果を利用したユニットテスト（正方形検出・ノイズ除去・空入力エラー）を作成。

## テスト
- 2025-11-09: `cargo test -- --test-threads=1`

