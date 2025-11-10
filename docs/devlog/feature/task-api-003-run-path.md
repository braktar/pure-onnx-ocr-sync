---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
end_date:
tags: [M3, api, pipeline]
depends_on:
---

# タスク概要
`OcrEngine::run_from_path` を実装し、画像パスからエンドツーエンドの OCR 推論を実行できるようにする。

## 要件
- ファイル読み込みと画像デコード処理を追加する
- 画像取得失敗時の明確なエラーを返却する
- 検出・認識結果をまとめて返却するレスポンス構造を定義する
- 擬似入力を用いた統合テストを追加する

## 実装メモ
- `OcrEngine::run_from_path` で画像ロードから検出・認識・CTC後処理までを連結し、`OcrResult`（テキスト・信頼度・ポリゴン）を返却。
- 検出パイプラインに `detect_polygons` を追加し、前処理・推論・後処理・スケーリングの各段階で詳細エラー (`OcrError`) を伝播。
- 認識パイプラインを `run` メソッドに集約し、矩形領域生成 (`RecTextRegion`) とバッチ推論／CTCデコードを統合。
- ダミー画像を用いた統合テストを新設し、ローカルモデル資産の有無を確認した上で `run_from_path` の成功を検証。

