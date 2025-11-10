---
status: todo
priority: medium
assignee: Backend
start_date:
end_date:
tags: [quality, metrics, recognition]
depends_on: task-fix-001
---

# タスク概要
CTC デコーダーの信頼度（confidence）計算を暫定仕様から脱却させ、PaddleOCR 準拠の Softmax 正規化に基づいた定量的スコアへ置き換える。

## 要件
- `ctc::DecodedSequence` が保持する `confidence` を Softmax ベースの平均対数尤度（もしくは同等の安定化手法）で算出し直すこと。
- 既存の `ocr_smoke` CLI で 0.000 固定になっている表示を、実測値に近いレンジへ更新する。
- ロジットのオーバーフローを避けるために log-sum-exp 等の数値安定化を必ず組み込む。
- 新アルゴリズムを検証するユニットテスト／統合テストを追加し、旧仕様との比較（最低限ベンチマークで 0 にならないこと）を確認する。
- タスク完了後、`README.md` や関連ドキュメントの「現在の制約」セクションを更新する。

