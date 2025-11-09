use crate::preprocessing::PreprocessedRecBatch;
use ndarray::Array3;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::anyhow::anyhow;

/// Result of running SVTR recognition inference.
#[derive(Debug, Clone)]
pub struct RecInferenceOutput {
    pub logits: Array3<f32>,
    pub valid_timesteps: Vec<usize>,
}

/// Runnable inference session for SVTR recognition model.
#[derive(Debug)]
pub struct RecInferenceSession {
    base_model: InferenceModel,
    cache: RefCell<HashMap<(usize, u32), Arc<TypedRunnableModel<TypedModel>>>>,
}

impl RecInferenceSession {
    pub fn load(model_path: impl AsRef<Path>) -> TractResult<Self> {
        let model_path = model_path.as_ref();
        println!("[RecInfer] Loading recognition model from {:?}", model_path);

        let mut inference_model = tract_onnx::onnx()
            .with_ignore_output_shapes(true)
            .model_for_path(model_path)?;

        let batch = inference_model.symbol_table.sym("batch");
        let width = inference_model.symbol_table.sym("width");
        inference_model.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![batch.into(), TDim::from(3), TDim::from(48), width.into()],
            ),
        )?;

        println!("[RecInfer] Recognition model prepared");
        Ok(Self {
            base_model: inference_model,
            cache: RefCell::new(HashMap::new()),
        })
    }

    pub fn run(&self, batch: &PreprocessedRecBatch) -> TractResult<RecInferenceOutput> {
        let tensor_shape = batch.tensor.shape();
        if tensor_shape.len() != 4 {
            return Err(anyhow!(
                "expected recognition input tensor to have 4 dimensions, got {:?}",
                tensor_shape
            )
            .into());
        }

        let batch_size = tensor_shape[0];
        let channel = tensor_shape[1];
        let height = tensor_shape[2];
        let width = tensor_shape[3];

        println!(
            "[RecInfer] Running inference with input shape {:?}",
            tensor_shape
        );

        if channel != 3 || height != 48 {
            return Err(anyhow!(
                "expected recognition input to have shape [*, 3, 48, *], got {:?}",
                tensor_shape
            )
            .into());
        }

        let plan = self.runnable_for_dims(batch_size, width as u32)?;
        let outputs = plan.run(tvec!(batch.tensor.clone().into()))?;
        let output_tensor = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("SVTR model did not return any outputs"))?;

        let view = output_tensor.to_array_view::<f32>()?;
        if view.ndim() != 3 {
            return Err(anyhow!(
                "expected recognition output to have 3 dimensions, got {:?}",
                view.shape()
            )
            .into());
        }

        let logits = view.into_dimensionality::<ndarray::Ix3>()?.to_owned();
        let (logit_batch, time_steps, _classes) = logits.dim();
        if logit_batch != batch_size {
            return Err(anyhow!(
                "batch dimension mismatch between input ({}) and output ({})",
                batch_size,
                logit_batch
            )
            .into());
        }

        let max_width = batch.max_width as f32;
        let scale = if max_width > 0.0 {
            time_steps as f32 / max_width
        } else {
            0.0
        };
        let valid_timesteps = batch
            .valid_widths
            .iter()
            .map(|width| {
                let mut steps = if scale > 0.0 {
                    (scale * *width as f32).round() as isize
                } else {
                    time_steps as isize
                };
                if steps < 1 {
                    steps = 1;
                }
                if steps as usize > time_steps {
                    steps = time_steps as isize;
                }
                steps as usize
            })
            .collect::<Vec<_>>();

        Ok(RecInferenceOutput {
            logits,
            valid_timesteps,
        })
    }

    fn runnable_for_dims(
        &self,
        batch_size: usize,
        width: u32,
    ) -> TractResult<Arc<TypedRunnableModel<TypedModel>>> {
        if let Some(plan) = self.cache.borrow().get(&(batch_size, width)) {
            return Ok(Arc::clone(plan));
        }

        println!(
            "[RecInfer] Preparing runnable model for batch {} width {}",
            batch_size, width
        );

        let mut model = self.base_model.clone();
        model.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![
                    TDim::from(batch_size as i64),
                    TDim::from(3),
                    TDim::from(48),
                    TDim::from(width as i64)
                ],
            ),
        )?;

        let plan = model
            .into_typed()?
            .into_decluttered()?
            .into_optimized()?
            .into_runnable()?;

        let plan = Arc::new(plan);
        self.cache
            .borrow_mut()
            .insert((batch_size, width), Arc::clone(&plan));

        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::{RecPreProcessor, RecPreProcessorConfig, RecTextRegion};
    use image::{DynamicImage, ImageBuffer, Rgb};
    use std::path::Path;

    fn gradient_image(width: u32, height: u32) -> DynamicImage {
        let mut buffer = ImageBuffer::new(width, height);
        for (x, y, pixel) in buffer.enumerate_pixels_mut() {
            let base = ((x + y) % 256) as u8;
            let green = base.saturating_add(16);
            let blue = base.saturating_add(32);
            *pixel = Rgb([base, green, blue]);
        }
        DynamicImage::ImageRgb8(buffer)
    }

    #[test]
    fn recognition_inference_runs() -> TractResult<()> {
        let model_path = Path::new("models/ppocrv5/rec.onnx");
        assert!(
            model_path.exists(),
            "expected SVTR model at {:?} to exist",
            model_path
        );

        let session = RecInferenceSession::load(model_path)?;

        let image = gradient_image(320, 160);
        let preprocessor = RecPreProcessor::new(RecPreProcessorConfig::default());
        let regions = vec![RecTextRegion {
            x: 10,
            y: 20,
            width: 120,
            height: 60,
        }];
        let batch = preprocessor
            .process(&image, &regions)
            .expect("recognition preprocessing should succeed");

        let output = session.run(&batch)?;
        let shape = output.logits.dim();

        assert_eq!(shape.0, 1);
        assert!(shape.1 > 0);
        assert!(shape.2 > 0);
        assert_eq!(output.valid_timesteps.len(), 1);
        assert!(output.valid_timesteps[0] <= shape.1);

        Ok(())
    }
}
