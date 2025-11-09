# Requirements Specification: Pure Rust OnnxOCR

Author: Shion Watanabe  
Date: 2025-11-09  
Repository: http://github.com/siska-tech/pure-onnx-ocr

## 1. Purpose & Background

- **Problem Statement**: Existing OnnxOCR implementations depend on Microsoft ONNX Runtime (C++). This complicates cross-compilation, WASM deployment, and embedded scenarios that prefer Rust-only stacks.
- **Goal**: Rebuild the full OCR pipeline in Pure Rust, delivering memory safety, predictable builds, and easier distribution.
- **Target Users**: Rust developers who need OCR without shipping native dependencies; teams deploying to serverless, embedded, or browser environments.

## 2. Functional Requirements

1. **Text Detection**  
   - Execute DBNet (`det.onnx`) to locate text polygons.  
   - Return polygons as `geo_types::Polygon`.
2. **Text Recognition**  
   - Crop each detected region, prepare input tensors, and run SVTR (`rec.onnx`).  
   - Decode logits into UTF-8 text strings.
3. **Integrated Facade**  
   - Provide a single API (`OcrEngine::run_from_path` / `run_from_image`) that orchestrates detection + recognition and returns `Vec<OcrResult>`.

### Inputs & Outputs

- **Inputs**: File path (`&str` / `AsRef<Path>`) or `image::DynamicImage`.
- **Outputs**: `Vec<OcrResult>` where each element holds `text`, `confidence`, `bounding_box`.

## 3. Non-Functional Requirements

### 3.1 Pure Rust Constraint

- No FFI to C/C++ libraries.
- Only use Rust crates that compile with the stable toolchain (e.g., `tract-onnx`, `ndarray`, `image`, `imageproc`, `i_overlay`, `geo-types`, optional `rayon`).

### 3.2 Performance

- Must execute the standard PP-OCRv5 ONNX models on CPU with acceptable latency (baseline: single-threaded inference comparable to Python + ONNX Runtime once `tract` optimizations complete).
- Recognition stage should leverage batching to amortize model execution costs.

### 3.3 Reliability & Risks

- `tract` operator coverage is an acknowledged risk; especially `LayerNormalization`, `Scan`, and dynamic shape handling. A PoC (completed) validated basic model loading, but upstream changes may still be required.
- Geometry operations must handle degenerate polygons gracefully—tests should cover near-collinear points, very small regions, and extreme aspect ratios.

### 3.4 Portability

- The library should compile on Windows, Linux, and macOS.
- WASM support is aspirational but not yet guaranteed; APIs must avoid blocking it (e.g., avoid spawning OS threads directly).

## 4. Constraints

| Category         | Constraint                                                         |
| ---------------- | ------------------------------------------------------------------ |
| Language         | Rust (stable channel), no nightly-only features in the public API. |
| Inference        | Use `tract-onnx` exclusively.                                      |
| Arrays           | Use `ndarray` for tensor manipulation; avoid `numpy` bindings.     |
| Image Processing | Use `image` + `imageproc`; avoid OpenCV bindings.                  |
| Geometry         | Use `i_overlay` + `geo-types`; avoid `pyclipper` or GEOS wrappers. |
| Licensing        | Prefer `MIT` / `Apache-2.0` compatible dependencies.               |

## 5. Success Criteria

- Full OCR pipeline runs end-to-end with PaddleOCR PP-OCRv5 models.
- Public API is documented, tested, and stable.
- README and design documents offer clear setup instructions for both Japanese and English-speaking contributors.
- Task `task-doc-001` delivers bilingual documentation assets to support the documentation milestone (M4).

## 6. Future Enhancements

- Optional GPU acceleration when Pure Rust backends mature.
- Export-friendly APIs (e.g., bindings for WASM or Python).
- Configurable decoding strategies (beam search, language models) once baseline greedy decoder is validated.

Refer to `docs/architecture_en.md` and `docs/detail_design_en.md` for deeper technical context, and `docs/test_specification_en.md` for validation scenarios.

