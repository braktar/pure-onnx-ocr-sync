---
status: todo
priority: low
assignee: Backend
start_date:
end_date:
tags: [benchmark, tooling, ocr-smoke]
depends_on: task-fix-000
---

# タスク概要
`ocr_smoke` CLI に推論時間および主要ステージ（前処理／推論／後処理）の計測を追加し、ベンチマーク用途に利用できるよう整備する。

## 要件
- 1 回の実行につき、全体の経過時間と少なくとも DBNet・SVTR 推論の時間を `info` ログとして出力する。
- 計測は `Instant` など標準 API を用いて行い、Windows・Unix 双方で動作すること。
- オプション（例: `--benchmark`）で計測のオン／オフを切り替え可能にし、既存の既定出力フォーマットを壊さない。
- `docs/devlog/task-fix-001-ocr-smoke-quality.md` 等で計測手順とサンプル出力を共有する。
- 将来的に自動ベンチマークへ取り込めるよう、測定結果を JSON 形式で保存する仕組みの追加も検討事項として記録する。

