# `pure-onnx-ocr`

Author: Shion Watanabe  
Date: 2025-11-09  
Repository: http://github.com/siska-tech/pure-onnx-ocr

Pure Rust OCR pipeline that re-implements the PaddleOCR (DBNet + SVTR\_HGNet) models without relying on C/C++ runtimes. The crate provides a high-level `OcrEngine` facade that hides detection and recognition stages behind a builder-style configuration API.

## Highlights

- **Pure Rust runtime** – no native shared libraries or FFI bindings; `cargo build` is enough.
- **DBNet + SVTR pipeline** – mirrors the official PaddleOCR ONNX export while staying within the Rust ecosystem.
- **Extensible architecture** – detection, recognition, and geometry utilities are separated so you can swap or extend individual stages.
- **Portable** – designed to run in environments where shipping C++ runtimes is difficult (embedded, serverless, WASM).

## Prerequisites

- Rust 1.75 or newer (stable channel)
- CPU inference on x86\_64 or aarch64
- ONNX models (`det.onnx`, `rec.onnx`) and the PaddleOCR dictionary (`ppocrv5_dict.txt`)

## Installation

```toml
[dependencies]
pure_onnx_ocr = "0.1.0"
image = "0.25"       # recommended for image I/O
geo-types = "0.7"    # recommended for working with polygon results
```

Download the `PP-OCRv5_Server-ONNX` (or Mobile) bundle from PaddleOCR. Place the files under `models/ppocrv5/` (or any path of your choice) and pass the paths into the builder.

## Quick Start

```rust
use pure_onnx_ocr::{OcrEngineBuilder, OcrResult};

fn main() -> Result<(), pure_onnx_ocr::OcrError> {
    let engine = OcrEngineBuilder::new()
        .det_model_path("models/ppocrv5/det.onnx")
        .rec_model_path("models/ppocrv5/rec.onnx")
        .dictionary_path("models/ppocrv5/ppocrv5_dict.txt")
        .det_limit_side_len(960)
        .det_unclip_ratio(1.5)
        .rec_batch_size(8)
        .build()?;

    let results: Vec<OcrResult> = engine.run_from_path("examples/demo.jpg")?;
    for (idx, result) in results.iter().enumerate() {
        println!(
            "#{} text={} confidence={:.4} polygon={:?}",
            idx,
            result.text,
            result.confidence,
            result.bounding_box.exterior().points()
        );
    }

    Ok(())
}
```

### Troubleshooting

- `ModelLoad`: `tract` rejected an operator that the ONNX graph requires (e.g., `LayerNormalization`, `Scan`). Try a simplified model or file an issue with model details.
- `Dictionary`: ensure the dictionary file is encoded in UTF-8 without BOM.

## API Overview

| Symbol             | Description                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `OcrEngineBuilder` | Configures model paths and runtime parameters. Produces an `OcrEngine`.                                       |
| `OcrEngine`        | Facade that executes detection + recognition. Provides `run_from_path` and `run_from_image`.                  |
| `OcrResult`        | Holds the text, confidence score, and `Polygon` bounding box for a single region.                             |
| `OcrError`         | Enumerates all errors emitted by the library (I/O, model loading, preprocessing, inference, post-processing). |
| `Polygon`          | Re-export of `geo-types::Polygon`. Useful for downstream geometry processing.                                 |

For detailed behavior and error semantics, see `docs/interface_design_en.md`.

## Documentation Set

- Architecture: `docs/architecture_en.md`
- Detailed design: `docs/detail_design_en.md`
- Interface design: `docs/interface_design_en.md`
- Requirements: `docs/requirements_en.md`
- References: `docs/references_en.md`
- Test specification: `docs/test_specification_en.md`

Each English document mirrors the Japanese source to help international contributors understand the project.

## Project Status

- 2025-11-09: Completed PoC for `det.onnx` (DBNet) loading via `tract-onnx`.
- 2025-11-09: Validated `rec.onnx` (SVTR\_HGNet) dummy inference; confirmed output shape `[1, 40, 18385]`.
- 2025-11-09: Implemented detection preprocessing (`DetPreProcessor`) with resizing, normalization, and NCHW transforms.
- 2025-11-09: Implemented detection inference session with runnable caching per input resolution.
- 2025-11-09: Implemented detection post-processing (contour extraction and filtering).
- 2025-11-09: Implemented polygon unclipping via `i_overlay` buffering.
- 2025-11-09: Implemented polygon scaling back to original coordinates.
- 2025-11-09: Implemented recognition preprocessing with cropping, force resize, normalization, and batching.
- 2025-11-09: Implemented recognition inference session with batch execution.
- 2025-11-09: Implemented dictionary loader with dedupe and bidirectional mapping.
- 2025-11-09: Implemented Pure Rust CTC greedy decoder with duplicate suppression and blank removal.
- 2025-11-09: Implemented recognition post-processor that combines logits, CTC decoding, and dictionary lookup.
- 2025-11-09: Implemented `OcrEngineBuilder`, `OcrEngine`, and public error surface.
- 2025-11-09: Refreshed README and added bilingual documentation set (`task-doc-001`).

## Contributing

Issues and pull requests are welcome. Please:

- Run `cargo fmt` and `cargo clippy` before submitting patches.
- Add unit tests where possible.
- Update the corresponding task file in `docs/devlog/` when documentation or feature work progresses.

## License

Licensed under `Apache-2.0`, aligning with PaddleOCR, OnnxOCR, and tract licensing.

