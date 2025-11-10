---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M3, api, pipeline]
depends_on:
---

# タスク概要
`OcrEngine::run_from_image` を実装し、メモリ上の画像バッファから OCR 推論を実行する。

## 要件
- 受け入れる画像バッファ形式とライフタイムを整理する
- 前処理とのインターフェース調整を行う
- エラー時の後始末と資源解放を保証する
- ユニットテストまたはベンチテストでシリアライズ済み画像を検証する

## 実装メモ
- `OcrEngine::run_from_image` を公開メソッドとして追加し、既存の `run_from_image_impl` を再利用。
- メモリ上の `DynamicImage` をそのまま検出パイプラインへ渡し、成功すれば `OcrResult` を返却するフローを `run_from_path` と共有。
- エラーハンドリングは `OcrError` へ統一し、前処理・推論・後処理段階で詳細情報を伝播。
- メモリ画像を利用したユニットテストを追加し、バッチサイズ制約内で結果が返ることを確認。

