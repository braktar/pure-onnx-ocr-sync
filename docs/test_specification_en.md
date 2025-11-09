# Test Specification: Pure Rust OnnxOCR

Author: Shion Watanabe  
Date: 2025-11-09  
Repository: http://github.com/siska-tech/pure-onnx-ocr

## Purpose

- Define the validation strategy based on the requirements, interface, and detailed design documents.
- De-risk the Pure Rust constraint by verifying `tract-onnx` compatibility with PP-OCRv5 models early.
- Ensure preprocessing and post-processing algorithms faithfully reproduce the behavior of the Python + OpenCV/Pyclipper reference.

## 1. Test Strategy

1. **Proof-of-Concept (PoC) Tests**  
   - Go/No-Go checks focused on model compatibility.  
   - Validate that `tract-onnx` can load and execute `rec.onnx` (SVTR\_HGNet) and `det.onnx` (DBNet). Failure implies the project cannot proceed without upstream changes.
2. **Unit Tests**  
   - Implemented with Rust’s built-in test harness.  
   - Cover detection/recognition preprocessing, tensor normalization, contour extraction, polygon buffering, dictionary loading, and CTC greedy decoding.
3. **Integration Tests**  
   - Placed under `tests/` and executed via `cargo test`.  
   - Exercise the public API (`OcrEngineBuilder`, `OcrEngine`) using real ONNX models and fixture images; verify end-to-end results.

**Coverage Goal**: ≥ 90% for core modules (pre/post-processing, decoding, dictionary).

## 2. Test Environment

- **Operating Systems**: Ubuntu 22.04+, macOS 12+, Windows 10/11.
- **Rust Toolchain**: Stable 1.75+.
- **Model Assets**: PP-OCRv5 Server ONNX bundle (`det.onnx`, `rec.onnx`, `ppocrv5_dict.txt`).
- **Image Fixtures**:
  - High-resolution natural scene with text.
  - Dense document image.
  - Blank image (negative test).
  - Corrupted image file (error path).

## 3. Test Cases

### 3.1 PoC Tests

| ID     | Scenario                                  | Expected Result                                                                               |
| ------ | ----------------------------------------- | --------------------------------------------------------------------------------------------- |
| PoC-01 | `OcrEngine::load_model` loads `rec.onnx`. | Returns `Ok(RunnableModel)`; otherwise `Err(OcrError::ModelLoad)` and the project is blocked. |
| PoC-02 | `OcrEngine::load_model` loads `det.onnx`. | Returns `Ok(RunnableModel)`.                                                                  |

### 3.2 Unit Tests

| Area                             | Focus                                                | Expected Outcome                                                        |
| -------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------- |
| Detection preprocess (resize)    | Landscape image with `det_limit_side_len = 960`.     | Output dims `(960, 540)`, `scale_ratio = 0.5`.                          |
| Detection preprocess (no resize) | Smaller input than limit.                            | Dimensions unchanged, `scale_ratio = 1.0`.                              |
| Detection tensor conversion      | Ensure NCHW layout and `[0, 1]` normalization.       | Tensor shape `(1, 3, H, W)` with clamped values.                        |
| Contour extraction               | Binary mask processed by `imageproc::find_contours`. | Expected number of contours returned.                                   |
| Polygon buffering                | Known polygon buffered with `i_overlay::buffering`.  | Offset polygon expands by configured ratio.                             |
| Coordinate rescaling             | Polygon and `scale_ratio` applied.                   | Coordinates restored to original image size.                            |
| Dictionary loader                | UTF-8 dictionary without duplicates.                 | Returns `(Vec<String>, blank_id)` where `blank_id == dictionary.len()`. |
| CTC greedy decode                | Handles repeats and blanks.                          | Collapses consecutive duplicates, removes blank token.                  |
| Recognition preprocess           | Force-resize cropped region to `(320, 48)`.          | Output tensor matches target dimensions.                                |
| Recognition batching             | 10 polygons, batch size 8.                           | Two batch tensors produced (8 + 2 items).                               |

### 3.3 Integration Tests

| Scenario                      | Input                                | Expected Result                                        |
| ----------------------------- | ------------------------------------ | ------------------------------------------------------ |
| Builder success path          | Valid model & dictionary paths.      | `Ok(OcrEngine)`                                        |
| Builder missing model         | Invalid detection model path.        | `Err(OcrError::Io)`                                    |
| Builder missing dictionary    | Dictionary path not found.           | `Err(OcrError::Io)`                                    |
| Builder model load failure    | Invalid/unsupported ONNX graph.      | `Err(OcrError::ModelLoad)`                             |
| `run_from_path` success       | Known image with text.               | Non-empty `Vec<OcrResult>` with correct text/polygons. |
| `run_from_image` success      | Same image loaded as `DynamicImage`. | Mirrors `run_from_path` result.                        |
| `run_from_path` no text       | Blank image.                         | Empty vector.                                          |
| `run_from_path` missing file  | Non-existent path.                   | `Err(OcrError::Io)`                                    |
| `run_from_path` corrupt image | Unreadable bytes.                    | `Err(OcrError::ImageDecode)`                           |

## 4. Automation & Tooling

- Test execution: `cargo test` (unit + integration).  
- Optional: `cargo nextest` for faster iteration (future work).
- CI should cache ONNX models to avoid repeated downloads.

## 5. Reporting

- Collect coverage via `cargo tarpaulin` (Linux) or `grcov` for multi-platform contexts.
- Record PoC results in `docs/devlog/` task logs to document model compatibility status.

## 6. Risk Mitigation

- Maintain minimal ONNX fixtures for CI to ensure legal redistribution.
- Document known unsupported operators with reproduction steps.
- Provide scripts to regenerate test assets if upstream models update.

Refer to `docs/requirements_en.md` for the rationale behind each test category and `docs/detail_design_en.md` for algorithmic details used in assertions.

