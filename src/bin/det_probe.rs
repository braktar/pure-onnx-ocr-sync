use pure_onnx_ocr_sync::run_dbnet_dummy_inference;
fn main() {
    let path = std::env::args().nth(1).expect("path");
    match run_dbnet_dummy_inference(&path) {
        Ok(o) => println!("OK: {} outputs", o.len()),
        Err(e) => eprintln!("FAIL: {}", e),
    }
}
