use std::path::Path;
use tract_onnx::prelude::*;

// サイズを小さくして試してみる
const DBNET_DUMMY_SHAPE: [usize; 4] = [1, 3, 320, 320]; // 小さいサイズに変更

pub fn run_dbnet_dummy_inference(model_path: impl AsRef<Path>) -> TractResult<TVec<Tensor>> {
    let model_path = model_path.as_ref();
    println!("Loading model from {:?}", model_path);

    let start = std::time::Instant::now();

    let mut model = tract_onnx::onnx()
        .with_ignore_output_shapes(true) // 出力シェイプの不一致を無視
        .model_for_path(model_path)?;
    println!("Model loaded, elapsed: {:?}", start.elapsed());

    let dummy_input: Tensor = tract_ndarray::Array4::<f32>::zeros(DBNET_DUMMY_SHAPE)
        .into_dyn()
        .into();

    model.set_input_fact(0, InferenceFact::from(&dummy_input))?;
    println!("Input fact set, elapsed: {:?}", start.elapsed());

    // 各ステップのタイミングをログに出力
    println!(
        "Starting model conversion to typed, elapsed: {:?}",
        start.elapsed()
    );
    let model = model.into_typed()?;

    println!("Starting decluttering, elapsed: {:?}", start.elapsed());
    let model = model.into_decluttered()?;

    println!("Starting optimization, elapsed: {:?}", start.elapsed());
    let model = model.into_optimized()?;

    println!("Making runnable, elapsed: {:?}", start.elapsed());
    let model = model.into_runnable()?;

    // タイムアウト機能は削除し、単純にログだけ残す
    println!("Total preparation time: {:?}", start.elapsed());

    println!("Running inference, elapsed: {:?}", start.elapsed());
    let outputs = model.run(tvec!(dummy_input.into()))?;
    println!("Inference complete, elapsed: {:?}", start.elapsed());

    Ok(outputs
        .into_iter()
        .map(|value| value.into_tensor())
        .collect::<TVec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn dbnet_dummy_inference_runs_successfully() -> TractResult<()> {
        let model_path = Path::new("models/ppocrv5/det.onnx");
        assert!(
            model_path.exists(),
            "expected DBNet model at {:?} to exist",
            model_path
        );

        println!("Starting inference test");
        let outputs = run_dbnet_dummy_inference(model_path)?;
        println!("Test completed with {} outputs", outputs.len());

        assert!(
            !outputs.is_empty(),
            "inference should return at least one output tensor"
        );

        // 出力のシェイプを表示
        for (i, tensor) in outputs.iter().enumerate() {
            println!("Output tensor #{} shape: {:?}", i, tensor.shape());
        }

        Ok(())
    }
}
