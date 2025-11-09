# `ROADMAP_fix.md`

## 🛠 Fix & Issue Backlog

The main roadmap tracks feature delivery.  
This document captures follow-up fixes, regression hunts, and operational hardening tasks.

### F0: Smoke Runner & Result Quality

| Status | Task ID              | Summary                                                                 | Notes                                                  |
| :----- | :------------------- | :---------------------------------------------------------------------- | :----------------------------------------------------- |
| `[x]`  | `task-doc-005`       | Ship `ocr_smoke` CLI and document current limitations                   | Baseline utility is ready; OCR result quality unstable |
| `[ ]`  | _TBD (issue pending)_ | Investigate noisy OCR outputs from `ocr_smoke` and stabilise detection | Blocked on analysis of DBNet/SVTR logits               |

### F1: Tooling & Diagnostics

| Status | Task ID        | Summary                                      | Notes |
| :----- | :------------- | :------------------------------------------- | :---- |
| `[ ]`  | _TBD_          | Add structured logging & tracing for engine | Draft once root-cause investigation starts |


