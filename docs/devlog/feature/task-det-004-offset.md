---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M1, postprocessing, unclip]
depends_on:
---

# タスク概要
抽出した輪郭ポリゴンに対し `i_overlay::buffering` を用いたオフセット (Unclip) 処理を実装する。

## 要件
- 輪郭から `geo-types::Polygon` へ変換する補助関数を実装する
- ポリゴンのサイズとアスペクト比に応じたバッファ距離の算出ロジックを設計する
- バッファ処理後の自己交差や縮退を検出し補正する
- 正常系・異常系のユニットテストを追加する

## 作業ログ
- 2025-11-09: `Cargo.toml` に `i_overlay = "4.1.1"` と `geo-types = "=0.7.12"` を追加し、Rust 1.70 互換なジオメトリ依存を導入。
- 2025-11-09: `src/postprocessing.rs` に `DetPolygonUnclipper` を実装。輪郭 -> `Polygon` 変換、DBNet の `area/perimeter` に基づくオフセット距離計算、`i_overlay` によるバッファ処理、結果フィルタを追加。
- 2025-11-09: 面積拡張を確認するユニットテスト `unclip_makes_polygon_larger` を追加し、閾値処理テストと合わせて検証。

## テスト
- 2025-11-09: `cargo test -- --test-threads=1`

