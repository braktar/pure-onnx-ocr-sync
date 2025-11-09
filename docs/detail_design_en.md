# Detailed Design: Pure Rust OnnxOCR

Author: Shion Watanabe  
Date: 2025-11-09  
Repository: http://github.com/siska-tech/pure-onnx-ocr

This document refines the architecture into concrete modules, data structures, and algorithms used to implement the OCR pipeline.

## 1. Internal Structures & Helpers

```rust
struct PreprocessedDetInput {
    tensor: tract_onnx::prelude::Tensor,
    resized_dims: (u32, u32),
    scale_ratio: f64,
}

struct PreprocessedRecInput {
    batch_tensor: tract_onnx::prelude::Tensor,
    original_polygons: Vec<geo_types::Polygon>,
}

struct OcrConfig {
    det_limit_side_len: u32,
    det_unclip_ratio: f64,
    det_thresh: f32,
    rec_image_shape: (u32, u32),
    rec_batch_size: usize,
}
```

`OcrEngine` owns the detection/recognition models, the dictionary, the blank token index, and the configuration. The builder validates paths, loads assets, and constructs the engine in a single step.

### Private Methods Inside `OcrEngine`

- `load_model(path) -> RunnableModel`: wraps `tract_onnx::onnx().model_for_path`.
- `load_dictionary(path) -> (Vec<String>, usize)`: reads UTF-8 dictionary files, deduplicates, and returns the blank index as `dictionary.len()`.
- `preprocess_detection(image) -> PreprocessedDetInput`: resizes while keeping aspect ratio, normalizes, converts to NCHW.
- `postprocess_detection(tensor, dims, ratio) -> Vec<Polygon>`: thresholds probability maps, extracts contours, buffers polygons, rescales to original coordinates.
- `preprocess_recognition(image, polygons) -> Vec<PreprocessedRecInput>`: crops each polygon, force-resizes to `(width, height)`, normalizes, stacks batches according to `rec_batch_size`.
- `postprocess_recognition(tensor) -> Vec<(String, f32)>`: `argmax` per timestep, remove duplicates/blanks, map to strings, compute confidences.

### Helper Functions

- `contour_to_polygon(contour) -> geo_types::Polygon`
- `ctc_greedy_decode(indices, blank_id) -> Vec<usize>`

## 2. Data Structures

| Structure          | Purpose                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------ |
| `OcrEngine`        | Holds models, dictionary, blank ID, configuration.                                                     |
| `OcrEngineBuilder` | Collects paths and configuration; validates before building.                                           |
| `OcrResult`        | Public struct with `text`, `confidence`, `bounding_box`.                                               |
| `OcrError`         | Public enum covering I/O, image decode, model load, inference, preprocessing/post-processing failures. |

## 3. Algorithms & Control Flow

### 3.1 `OcrEngineBuilder::build`

1. Ensure detection, recognition, and dictionary paths are provided.
2. Load detection and recognition models via `load_model`.
3. Load dictionary via `load_dictionary`.
4. Construct `OcrEngine` with the loaded artifacts.

### 3.2 Detection Pipeline

1. **Preprocess**:
   - Compute scale ratio based on `det_limit_side_len`.
   - Resize using `image::imageops::resize`.
   - Normalize pixels to `[0, 1]`, convert to NCHW via `ndarray`.
2. **Inference**:
   - Convert tensor to `tract` input `Datum`.
   - Cache `RunnableModel` per `(width, height)` combination.
3. **Postprocess**:
   - Apply threshold (e.g., `det_thresh = 0.3`).
   - Use `imageproc::contours::find_contours` to obtain polygons.
   - Filter by area and number of points.
   - Buffer polygons using `i_overlay::buffering` with `det_unclip_ratio`.
   - Rescale coordinates back to original image size.

### 3.3 Recognition Pipeline

1. **Preprocess**:
   - For each polygon, crop the corresponding quad from the original image.
   - Force resize to `(rec_image_shape.0, rec_image_shape.1)`.
   - Normalize RGB channels to `[0, 1]`.
   - Stack crops into batches of `rec_batch_size` using `ndarray::stack`.
2. **Inference**:
   - Feed batch tensor to `tract`.
   - Support dynamic batch sizes (pad the final batch if needed).
3. **Postprocess**:
   - Convert logits to `ndarray::Array3<f32>`.
   - Run `argmax` along the class axis.
   - Apply CTC greedy decoding to remove duplicates and blanks.
   - Map indices to string tokens using the loaded dictionary.
   - Compute confidence via averaged max probabilities.

## 4. Error Handling

- All internal errors map to `OcrError` variants:
  - `Io` for file access issues.
  - `ModelLoad` when `tract` cannot parse or optimize an ONNX graph.
  - `Dictionary` for malformed dictionaries (duplicates, UTF-8 errors).
  - `DetectionPreprocess`, `DetectionInference`, `DetectionPostProcess`.
  - `RecognitionPreprocess`, `RecognitionInference`, `RecognitionPostProcess`.
  - `PipelineMismatch` if the number of polygons and decoded strings differ (should not happen if each stage succeeds).

## 5. Testing Strategy

- Unit tests cover:
  - Image resizing and normalization edge cases (portrait vs landscape).
  - Polygon buffering robustness across different aspect ratios.
  - CTC decoder behavior with duplicates and blank IDs.
  - Dictionary loader handling of BOMs and duplicate entries.
- Integration tests (planned) will load lightweight ONNX fixtures to validate the full pipeline with deterministic inputs.

## 6. Performance Considerations

- Cache `RunnableModel` instances per input shape to avoid recompilation.
- Optionally compile with `rayon` to parallelize recognition preprocessing.
- Avoid unnecessary tensor allocations by reusing buffers inside preprocessors.

## 7. Future Work

- Investigate quantized ONNX models to reduce inference latency.
- Add GPU-backed inference once `tract` exposes stable CUDA/WGPU support.
- Support rotated or curved text detection by integrating geometry utilities beyond simple polygons.

## References

See `docs/references_en.md` for the bibliography and `docs/test_specification_en.md` for planned integration scenarios.

