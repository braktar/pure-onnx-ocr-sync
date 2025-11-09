# Interface Design (API Specification): Pure Rust OnnxOCR

Author: Shion Watanabe  
Date: 2025-11-09  
Repository: http://github.com/siska-tech/pure-onnx-ocr

## Purpose

- Describe the public API exposed by the crate.
- Ensure application developers can use the OCR pipeline safely and intuitively.
- Align with the architectural decision to provide a Facade via `OcrEngine`.

## Module Layout

```
pure_onnx_ocr
 ├─ OcrEngine
 ├─ OcrEngineBuilder
 ├─ OcrResult
 ├─ OcrError
 └─ re-export: geo_types::{Point, Polygon}
```

## Public API Summary

### `OcrEngineBuilder`

| Method                                                   | Description                                                      |
| -------------------------------------------------------- | ---------------------------------------------------------------- |
| `new() -> Self`                                          | Create a builder with default configuration.                     |
| `det_model_path<P: AsRef<Path>>(self, path: P) -> Self`  | Set the DBNet ONNX file path.                                    |
| `rec_model_path<P: AsRef<Path>>(self, path: P) -> Self`  | Set the SVTR ONNX file path.                                     |
| `dictionary_path<P: AsRef<Path>>(self, path: P) -> Self` | Set the CTC dictionary text file path.                           |
| `det_limit_side_len(self, len: u32) -> Self`             | Optional: limit the longest side during detection preprocessing. |
| `det_unclip_ratio(self, ratio: f64) -> Self`             | Optional: polygon offset ratio for DBNet unclip logic.           |
| `rec_batch_size(self, size: usize) -> Self`              | Optional: batch size for recognition inference.                  |
| `build(self) -> Result<OcrEngine, OcrError>`             | Validate configuration, load assets, and produce an engine.      |

Builder methods consume `self` to encourage method chaining and prevent partially configured instances from being reused.

### `OcrEngine`

| Method                                               | Description                                                                        |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `run_from_path<P: AsRef<Path>>(&self, path: P)`      | Load an image from disk, run detection + recognition, and return `Vec<OcrResult>`. |
| `run_from_image(&self, image: &image::DynamicImage)` | Run OCR on an in-memory image buffer.                                              |

Both methods return `Result<Vec<OcrResult>, OcrError>` and are thread-safe. The engine holds immutable references to models and dictionary, enabling concurrent calls from multiple threads.

### `OcrResult`

```rust
pub struct OcrResult {
    pub text: String,
    pub confidence: f32,
    pub bounding_box: Polygon,
}
```

- `text`: UTF-8 string decoded by the CTC pipeline.
- `confidence`: Average of maximum probabilities per time step (0.0–1.0).
- `bounding_box`: `geo_types::Polygon` describing the detected region in original image coordinates.

### `OcrError`

```rust
pub enum OcrError {
    MissingField { field: &'static str },
    Io { path: PathBuf, source: std::io::Error },
    ModelLoad { path: PathBuf, source: anyhow::Error },
    Dictionary { source: anyhow::Error },
    InvalidConfiguration { message: String },
    ImageDecode { path: PathBuf, source: image::ImageError },
    DetectionPreprocess { source: anyhow::Error },
    DetectionInference { source: anyhow::Error },
    DetectionPostProcess { source: anyhow::Error },
    RecognitionPreprocess { source: anyhow::Error },
    RecognitionInference { source: anyhow::Error },
    RecognitionPostProcess { source: anyhow::Error },
    PipelineMismatch { detection_regions: usize, recognition_results: usize },
}
```

- Implements `std::error::Error` and `Display`.
- `MissingField` is raised by the builder when required paths are absent.
- `ModelLoad` wraps `tract` parsing errors (including unsupported operators).
- Post-processing errors wrap geometry-related issues (invalid polygons, buffering failures).

### Re-exports

The crate re-exports `geo_types::{Point, Polygon}` so downstream users do not need to depend on `geo-types` directly when consuming OCR results.

## Usage Patterns

- Instantiate the builder once during application startup, keep the resulting `OcrEngine` in a shared context (e.g., `Arc<OcrEngine>`).
- Use `run_from_path` for file-based batch processing utilities.
- Use `run_from_image` when the image is already loaded in memory (e.g., REST API receiving bytes).

## Error Handling Guidelines

- Applications should inspect `OcrError` variants to differentiate between transient I/O issues and permanent model incompatibilities.
- Logging should include the `path` fields when present to help diagnose deployment problems.
- For `ModelLoad` errors caused by unsupported operators, provide guidance to users about model conversion or simplification.

## Extensibility

- Additional configuration setters may be added to the builder (e.g., confidence thresholds) without breaking existing code.
- Feature flags can enable optional dependencies such as `rayon` for parallelism or `serde` for serializing OCR results.

## Testing Expectations

- Public APIs are covered by integration tests that use fixture images and lightweight ONNX models.
- Each error variant is exercised by dedicated tests to ensure the correct variant and message are emitted.

## Related Documents

- Architecture overview: `docs/architecture_en.md`
- Detailed design: `docs/detail_design_en.md`
- Requirements specification: `docs/requirements_en.md`

