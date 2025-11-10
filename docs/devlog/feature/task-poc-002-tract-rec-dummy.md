---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-09
end_date: 2025-11-09
tags: [M0, tract, svtr]
depends_on:
---

# タスク概要
`tract-onnx` で `rec.onnx` (SVTR) モデルをロードし、簡易的なダミー入力で推論を通せることを検証する。

## 要件
- SVTR の入力サイズに合わせたダミー画像テンソルを生成する
- `tract-onnx` を用いてモデルをロードし推論可能なパイプラインを構築する
- 推論後の出力テンソル形状と数値範囲を記録する
- 検証結果と想定される次の課題を開発ログにまとめる

## 実行メモ (2025-11-09)

- `cargo test svtr_dummy_inference_runs_successfully -- --nocapture` を実行。
- 入力テンソルは `SVTR_DUMMY_SHAPE = [1, 3, 48, 320]` の NCHW 配列。チャンネルごとに位相をずらしたサイン波グラデーションを生成して正規化済みの疑似画像を模倣。
- モデル読み込み直後の型推論・最適化 (`into_decluttered`) に **約 6 分 (≈ 362s)** を要し、推論全体のボトルネックになっている。
- 1 バッチ推論の出力テンソル形状は **`[1, 40, 18385]`**。時系列長 40、語彙サイズ 18,385（`ppocrv5_dict.txt` のトークン数と一致）を確認。
- 出力値の範囲は **min = 0.0、max ≈ 0.981465**。`softmax` 相当の正規化が行われ、確率分布として扱えることを確認。

## 想定される次の課題

- `into_decluttered` 段階での極端な遅延を解消するため、`tract` の最適化パイプラインの調査、もしくは事前最適化済みモデルのキャッシュ戦略を検討する。
- 現状は単発推論のみ。バッチ推論と `CTC` 後処理を接続するために、`RecPreProcessor` (`task-rec-001`) と `CTC Greedy Decode` (`task-rec-004`) の実装が必要。
- 18,385 クラスの出力を文字列へマッピングする辞書ローダ (`task-rec-003`) と統合ユニットテストの整備が未着手。

