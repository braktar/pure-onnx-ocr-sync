---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-10
tags: [M4, tests, integration]
depends_on:
---

# タスク概要
`tests/integration_test.rs` を拡充し、検出・認識パイプラインの結合テストを整備する。

## 要件
- テストデータの配置方針とフィクスチャを設計する
- 成功・失敗ケースを含むシナリオテストを実装する
- テストの並列実行時に競合しないよう資源管理を実装する
- CI での実行手順と所要時間を文書化する

## 作業ログ
- `tests/integration_test.rs` を作成し、フィクスチャ存在時のみ実行されるパイプライン結合テストを実装
- `PURE_ONNX_OCR_FIXTURE_DIR` を通じたモデル・画像共有戦略と `OnceLock` によるリソース共有で並列安全性を確保
- テスト用フィクスチャ構成 (`tests/fixtures/README.md`) と実行手順を README に追記

