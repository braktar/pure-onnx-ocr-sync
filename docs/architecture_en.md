# Architecture Specification: Pure Rust OnnxOCR

Author: Shion Watanabe  
Date: 2025-11-09  
Repository: http://github.com/siska-tech/pure-onnx-ocr

## Purpose

- Define the top-level module structure of the OCR pipeline.
- Clarify responsibilities and dependencies among detection, recognition, and shared utilities.
- Capture the architectural principles that guide the Pure Rust implementation.

## System Overview

```
    +-------------------+
    | Application / CLI |
    +---------+---------+
              |
              v
    +-------------------+
    |  OcrEngine (API)  |
    +---------+---------+
        |             |
        v             v
+---------------+ +---------------+
| Detection     | | Recognition   |
| Pipeline      | | Pipeline      |
+-------+-------+ +-------+-------+
        |                 |
        v                 v
  Preprocess        Preprocess
        |                 |
        v                 v
    Inference         Inference
        |                 |
        v                 v
   Postprocess       Postprocess
        |                 |
        +--------+--------+
                 |
                 v
         Vec<OcrResult>
```

### Key Modules

| Module                | Responsibility                                                 | Notes                                           |
| --------------------- | -------------------------------------------------------------- | ----------------------------------------------- |
| `OcrEngine`           | Orchestrates the full pipeline and exposes the public API.     | Implements the Facade pattern.                  |
| `DetPreProcessor`     | Resize, normalize, and convert images to NCHW tensors.         | Produces `tract::Tensor` + scale metadata.      |
| `DetInferenceSession` | Run `det.onnx` DBNet model with `tract-onnx`.                  | Caches runnable instances by input size.        |
| `DetPostProcessor`    | Extract contours, unclip polygons, rescale to original space.  | Uses `imageproc`, `i_overlay`, and `geo-types`. |
| `RecPreProcessor`     | Crop polygons, force-resize, normalize, batch for recognition. | Returns batches as `tract::Tensor`.             |
| `RecInferenceSession` | Run `rec.onnx` SVTR model with `tract-onnx`.                   | Handles dynamic batch sizes.                    |
| `RecPostProcessor`    | Apply argmax, CTC greedy decoding, dictionary lookup.          | Computes text + confidence scores.              |

## Data Flow

1. `OcrEngine::run` receives a file path or `DynamicImage`.
2. Detection preprocessing resizes the image (limit-side length) and produces an input tensor.
3. Detection inference generates probability maps.
4. Detection post-processing thresholds maps, finds contours, performs polygon buffering, and rescales coordinates.
5. Recognition preprocessing crops each polygon, builds fixed-size tensors, and stacks batches.
6. Recognition inference returns logits for each time step.
7. Recognition post-processing applies argmax, removes blanks/duplicates, and maps indices to strings.
8. `OcrEngine` combines polygons + text into `Vec<OcrResult>` and returns it to the caller.

## Architectural Principles

- **Separation of Concerns** – preprocessing, inference, and post-processing live in distinct modules. Detection and recognition are fully decoupled.
- **Pure Rust** – replace all C/C++ tooling with Rust crates: `tract-onnx`, `image`, `imageproc`, `i_overlay`, `geo-types`, `ndarray`, optionally `rayon`.
- **Facade Pattern** – `OcrEngine` hides the complexity of the pipeline behind a minimal API surface.
- **Builder Pattern** – `OcrEngineBuilder` collects model paths and runtime options, preventing partial initialization.

## Technology Choices

| Area                   | Crate        | Reason                                                        |
| ---------------------- | ------------ | ------------------------------------------------------------- |
| ONNX inference         | `tract-onnx` | Pure Rust inference engine compatible with DBNet/SVTR graphs. |
| Image IO & resize      | `image`      | Handles decoding and resizing without native dependencies.    |
| Contour detection      | `imageproc`  | Pure Rust alternative to OpenCV `findContours`.               |
| Polygon buffering      | `i_overlay`  | Pure Rust replacement for `pyclipper` polygon offsetting.     |
| Geometry types         | `geo-types`  | Provides polygon primitives for downstream consumers.         |
| Tensor utilities       | `ndarray`    | Manipulates tensors (permute axes, stack, argmax) in Rust.    |
| Parallelism (optional) | `rayon`      | Enables batch-level parallel execution when needed.           |

## Risks & Mitigations

- **Operator Support** – `rec.onnx` relies on advanced operators (`LayerNormalization`, `Scan`). Mitigation: validate models early and track upstream `tract` support.
- **Performance** – CPU-only inference may be slower than native PaddleOCR. Mitigation: batch recognition, cache runnables, and profile hotspots.
- **Geometry Precision** – polygon offsetting and scaling must remain numerically stable. Mitigation: rely on `geo-types` and thorough unit tests.

## References

- Architecture diagrams and module details: `docs/detail_design_en.md`
- API surface: `docs/interface_design_en.md`
- Requirements and constraints: `docs/requirements_en.md`

