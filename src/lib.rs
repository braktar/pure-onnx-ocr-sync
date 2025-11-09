pub mod ctc;
pub mod detection;
pub mod dictionary;
pub mod engine;
pub mod postprocessing;
pub mod preprocessing;
pub mod recognition;

pub use ctc::{CtcGreedyDecoder, CtcGreedyDecoderConfig, CtcGreedyDecoderError, DecodedSequence};
pub use detection::{DetInferenceOutput, DetInferenceSession};
pub use dictionary::{DictionaryError, RecDictionary};
pub use engine::{OcrEngine, OcrEngineBuilder, OcrEngineConfig, OcrError, OcrResult};
pub use geo_types::{Point, Polygon};
pub use postprocessing::{
    DetPolygonScaler, DetPolygonScalerConfig, DetPolygonUnclipper, DetPolygonUnclipperConfig,
    DetPostProcessor, DetPostProcessorConfig, DetPostProcessorError, DetScaleRounding,
    DetUnclipLineJoin,
};
pub use preprocessing::{
    DetPreProcessor, DetPreProcessorConfig, DetPreProcessorError, PreprocessedDetInput,
    PreprocessedRecBatch, RecPreProcessor, RecPreProcessorConfig, RecPreProcessorError,
    RecTextRegion,
};
pub use recognition::{
    RecInferenceOutput, RecInferenceSession, RecPostProcessor, RecPostProcessorConfig,
    RecPostProcessorError,
};

use std::path::Path;
use tract_onnx::prelude::*;

const DBNET_DUMMY_SHAPE: [usize; 4] = [1, 3, 320, 320];
const SVTR_DUMMY_SHAPE: [usize; 4] = [1, 3, 48, 320];

pub fn run_dbnet_dummy_inference(model_path: impl AsRef<Path>) -> TractResult<TVec<Tensor>> {
    let dummy_input: Tensor = tract_ndarray::Array4::<f32>::zeros(DBNET_DUMMY_SHAPE)
        .into_dyn()
        .into();
    run_dummy_inference(model_path, dummy_input, "DBNet")
}

pub fn run_svtr_dummy_inference(model_path: impl AsRef<Path>) -> TractResult<TVec<Tensor>> {
    let dummy_input: Tensor =
        tract_ndarray::Array4::<f32>::from_shape_fn(SVTR_DUMMY_SHAPE, |(_, channel, row, col)| {
            // 正規化された斜めグラデーション: チャンネルごとにスケールを変えて変化を持たせる
            let spatial_size = (SVTR_DUMMY_SHAPE[2] * SVTR_DUMMY_SHAPE[3]) as f32;
            let base = (row * SVTR_DUMMY_SHAPE[3] + col) as f32 / spatial_size;
            let channel_scale = 0.1 * channel as f32;
            (base + channel_scale).sin()
        })
        .into_dyn()
        .into();
    run_dummy_inference(model_path, dummy_input, "SVTR")
}

fn run_dummy_inference(
    model_path: impl AsRef<Path>,
    dummy_input: Tensor,
    label: &str,
) -> TractResult<TVec<Tensor>> {
    let model_path = model_path.as_ref();
    println!("[{}] Loading model from {:?}", label, model_path);

    let start = std::time::Instant::now();

    let mut model = tract_onnx::onnx()
        .with_ignore_output_shapes(true)
        .model_for_path(model_path)?;
    println!("[{}] Model loaded, elapsed: {:?}", label, start.elapsed());

    model.set_input_fact(0, InferenceFact::from(&dummy_input))?;
    println!("[{}] Input fact set, elapsed: {:?}", label, start.elapsed());

    println!(
        "[{}] Starting model conversion to typed, elapsed: {:?}",
        label,
        start.elapsed()
    );
    let model = model.into_typed()?;

    println!(
        "[{}] Starting decluttering, elapsed: {:?}",
        label,
        start.elapsed()
    );
    let model = model.into_decluttered()?;

    println!(
        "[{}] Starting optimization, elapsed: {:?}",
        label,
        start.elapsed()
    );
    let model = model.into_optimized()?;

    println!(
        "[{}] Making runnable, elapsed: {:?}",
        label,
        start.elapsed()
    );
    let model = model.into_runnable()?;

    println!("[{}] Total preparation time: {:?}", label, start.elapsed());

    println!(
        "[{}] Running inference, elapsed: {:?}",
        label,
        start.elapsed()
    );
    let outputs = model.run(tvec!(dummy_input.into()))?;
    println!(
        "[{}] Inference complete, elapsed: {:?}",
        label,
        start.elapsed()
    );

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
    #[ignore = "dummy inference takes >60s; run with `cargo test -- --ignored`"]
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

    #[test]
    #[ignore = "dummy inference takes >60s; run with `cargo test -- --ignored`"]
    fn svtr_dummy_inference_runs_successfully() -> TractResult<()> {
        let model_path = Path::new("models/ppocrv5/rec.onnx");
        assert!(
            model_path.exists(),
            "expected SVTR model at {:?} to exist",
            model_path
        );

        println!("Starting SVTR inference test");
        let outputs = run_svtr_dummy_inference(model_path)?;
        println!("SVTR test completed with {} outputs", outputs.len());

        assert!(
            !outputs.is_empty(),
            "SVTR inference should return at least one output tensor"
        );

        let first = &outputs[0];
        println!("SVTR output tensor shape: {:?}", first.shape());

        let shape = first.shape().to_vec();
        assert_eq!(
            shape.first().copied(),
            Some(1),
            "SVTR batch dimension should be 1"
        );
        assert!(
            shape.iter().skip(1).all(|dim| *dim > 0),
            "SVTR tensor dimensions after batch should be positive"
        );

        let view = first.to_array_view::<f32>()?;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for value in view.iter() {
            min = min.min(*value);
            max = max.max(*value);
        }
        println!("SVTR output value range: min={:.6}, max={:.6}", min, max);

        assert!(
            min.is_finite() && max.is_finite(),
            "SVTR output values should be finite numbers"
        );
        assert!(
            max > min,
            "SVTR output values should have a non-zero dynamic range"
        );

        Ok(())
    }
}
