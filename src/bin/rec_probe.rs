use pure_onnx_ocr_sync::run_svtr_dummy_inference;
use std::path::PathBuf;
fn main() {
    let path = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("examples/web/dist/models/ocr/PP-OCRv5_server_rec_infer.onnx")
    });
    match run_svtr_dummy_inference(&path) {
        Ok(o) => println!("OK: {} outputs, shape {:?}", o.len(), o[0].shape()),
        Err(e) => eprintln!("FAIL: {}", e),
    }
}
