---
status: progress
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date:
tags: [M1, postprocessing, scaling]
depends_on:
---

# タスク概要
オフセット済みポリゴンを元画像座標にスケールバックし、`geo-types::Polygon` のリストとして出力する。

## 要件
- リサイズ倍率とパディング量を考慮した座標逆変換を実装する
- 線形補間誤差を最小化するための丸め戦略を定義する
- 出力ポリゴンの順序と一貫性を保証する
- 単体テストで複数解像度の入力ケースを検証する

## 作業ログ
- 2025-11-09: `DetPolygonScaler` を `src/postprocessing.rs` に追加。縮小済みポリゴンを元解像度へ逆スケーリングし、丸め・クリッピングを行うロジックを実装。
- 2025-11-09: 代表的ケース（縮小→復元、境界超過のクリッピング）を確認するユニットテストを追加。

## テスト
- 2025-11-09: `cargo test -- --test-threads=1`

