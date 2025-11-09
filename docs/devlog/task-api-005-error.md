---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M3, api, error]
depends_on:
---

# タスク概要
公開エラー型 `OcrError` を定義し、パイプライン全体の失敗を一元的に扱えるようにする。

## 要件
- エラーカテゴリとメッセージ構造を設計する
- 主要コンポーネントから `OcrError` への変換を実装する
- `std::error::Error` と `Display` 実装を提供する
- エラー伝播のユニットテストを追加する

## 実装メモ
- `DetPreProcessorError` / `DetPostProcessorError` / `RecPreProcessorError` / `RecPostProcessorError` から `OcrError` への `From` 実装を追加し、`map_err(OcrError::from)` でステージを区別したままエラー伝播できるようにした。
- 検出・認識パイプラインの `process` / `run` 呼び出しを簡素化し、`?` 記法で `OcrError` を返す構造に整理。
- コンポーネントエラーの変換結果を検証するユニットテスト `component_errors_convert_to_ocr_error_variants` を追加し、意図したバリアントにマッピングされることを確認。

