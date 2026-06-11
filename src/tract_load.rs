use std::io::Cursor;

use tract_onnx::prelude::*;

/// ONNX loader tuned for Paddle-exported graphs (PP-OCRv5).
pub fn paddle_onnx() -> tract_onnx::model::Onnx {
    tract_onnx::onnx().with_ignore_output_shapes(true)
}

/// Load an ONNX model protobuf from an in-memory buffer (browser WASM / embedded hosts).
pub fn inference_model_from_bytes(bytes: &[u8]) -> TractResult<InferenceModel> {
    let mut cursor = Cursor::new(bytes);
    paddle_onnx().model_for_read(&mut cursor)
}
