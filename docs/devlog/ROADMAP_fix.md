# `ROADMAP_fix.md`

## 🛠 Fix & Issue Backlog

The main roadmap tracks feature delivery.  
This document captures follow-up fixes, regression hunts, and operational hardening tasks.

### F0: Smoke Runner & Result Quality

| Status | Task ID              | Summary                                                                 | Notes                                                  |
| :----- | :------------------- | :---------------------------------------------------------------------- | :----------------------------------------------------- |
| `[x]`  | `task-fix-000`       | Ship `ocr_smoke` CLI and document current limitations                   | Baseline utility is ready; OCR result quality unstable |
| `[ ]`  | `task-fix-001`        | Investigate noisy OCR outputs from `ocr_smoke` and stabilise detection | Branch `fix/001-ocr-smoke-quality`; blocked on analysis of DBNet/SVTR logits |

### F1: Tooling & Diagnostics

| Status | Task ID        | Summary                                      | Notes |
| :----- | :------------- | :------------------------------------------- | :---- |
| `[ ]`  | _TBD_          | Add structured logging & tracing for engine | Draft once root-cause investigation starts |


