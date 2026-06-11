# lib-know patches (vs upstream `pure-onnx-ocr-sync` 0.2.0)

Base commit: [zhangtao103239/pure-onnx-ocr-sync](https://github.com/zhangtao103239/pure-onnx-ocr-sync) @ `299d11c` (crates.io **0.2.0**).

These changes support **lib-know** (WASM/mobile live camera, in-memory ONNX assets, and robust DBNet on blurry frames).

## Cargo.toml

- Add `web-time` dependency on `wasm32` (monotonic clock where `std::time::Instant` is unavailable).

## New modules

- **`src/tract_load.rs`** — shared Paddle ONNX loader (`with_ignore_output_shapes(true)`) and `inference_model_from_bytes` for in-memory models.
- **`src/timer.rs`** — `Instant` alias: `web_time::Instant` on `wasm32`, `std::time::Instant` elsewhere.

## ONNX loading (path + bytes)

- **`detection.rs`**, **`recognition.rs`**, **`text_line_ori.rs`**, **`doc_ori.rs`**: `load_from_bytes`, refactor path loading through `tract_load::paddle_onnx()`, shared `prepare_*` helpers.
- **`dictionary.rs`**: `from_utf8_bytes` / shared `parse_contents` for in-memory dictionaries.
- **`lib.rs`**: use `tract_load` + `timer`; shorten dummy SVTR warmup (`into_typed()?.into_runnable()` only); export `OnnxModelBytes`, `central_address_region`.

## Engine / pipeline (`engine.rs`)

- **`OnnxModelBytes`** and **`OcrEngineBuilder::build_from_bytes`** for hosts without a filesystem.
- **`set_fast_detection` / `is_fast_detection`**: skip doc-orientation and use a single 320px DBNet scale on WASM/mobile.
- **`detect_text_regions`**: public DBNet-only API with multi-scale retry and contrast enhancement fallback.
- **`detect_polygons_multi_scale`**, **`orient_image_for_detection`**: multi long-side limits (960→320) when tract graph compile or empty detection; fast path skips doc-ori.
- **`central_address_region`**, **`address_line_regions`**: heuristic central band when DBNet finds no boxes (aligned with lib-know label geometry).
- **`enhance_for_detection`**: per-channel contrast stretch retry for blurry webcam frames.
- Full pipeline uses multi-scale detection and falls back to **4 horizontal line regions** in the central band when no polygons are detected.

## Detection preprocessing (`preprocessing.rs`)

- **`process_with_limit`**, **`limit_side_len`**, **`detection_tensor_dims`**: explicit long-side limit for multi-scale DBNet; tensor sizing changes vs upstream padding-only resize.

## Dev / debug binaries

- **`src/bin/det_probe.rs`**, **`src/bin/rec_probe.rs`**: probe helpers (not in upstream).

## Tests (`lib.rs`)

- **`mobile_svtr_dummy_inference_runs`**: optional ignored test against lib-know mobile rec ONNX path.

## Metadata (this fork)

- `repository` / `homepage` point to `https://github.com/braktar/pure-onnx-ocr-sync` on branch **`lib-know`**.
