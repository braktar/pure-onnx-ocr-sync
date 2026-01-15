use crate::preprocessing::Rotation;
use crate::{OcrError, StageTimings};
use image::imageops::FilterType::Triangle;
use image::{DynamicImage};
use std::path::Path;
use std::time::{Duration, Instant};
use tract_onnx::prelude::*;
use tract_onnx::tract_core::anyhow::anyhow;

#[derive(Debug)]
pub struct DocOriInferenceSession {
    base_model: TypedRunnableModel<TypedModel>,
    height: i32,
    width: i32,
}

impl DocOriInferenceSession {
    pub fn load(model_path: impl AsRef<Path>) -> TractResult<Self> {
        let model_path = model_path.as_ref();
        println!(
            "[DocOriInfer] Loading PP-LCNet_x1_0_doc_ori model from {:?}",
            model_path
        );
        let height = 224;
        let width = 224;
        let mut inference_model = tract_onnx::onnx()
            .with_ignore_output_shapes(true)
            .model_for_path(model_path)?;
        let batch_sym = inference_model.symbol_table.sym("DynamicDimension.0");
        inference_model.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![
                    batch_sym.into(),
                    TDim::from(3),
                    TDim::from(height),
                    TDim::from(width)
                ],
            ),
        )?;
        let runnable_model = inference_model
            .into_typed()?
            .into_decluttered()?
            .into_optimized()?
            .into_runnable()?;

        println!("[DocOriInfer] Model PP-LCNet_x1_0_doc_ori prepared");
        Ok(Self {
            base_model: runnable_model,
            height: height,
            width: width,
        })
    }

    pub fn run(
        &self,
        images: Vec<DynamicImage>,
    ) -> TractResult<Vec<Rotation>> {
        let batch_size = images.len();

        println!(
            "[DocOriInfer] Running inference with input dims {:?}",
            batch_size
        );

        let (width, height) = (self.width as u32, self.height as u32);
        // ImageNet 均值和标准差
        let mean = [0.485, 0.456, 0.406]; // R, G, B
        let std = [0.229, 0.224, 0.225]; // R, G, B
        let mut rotations = Vec::with_capacity(batch_size);
        let mut tensor = Vec::with_capacity(((batch_size as u32) * 3 * height * width) as usize);
        for i in 0..batch_size {
            let img = &images[i];
            let rgb_img = img.resize_exact(width, height, Triangle).to_rgb8();
            for channel in 0..3 {
                for y in 0..height {
                    for x in 0..width {
                        let pixel = rgb_img.get_pixel(x, y);
                        tensor.push(
                            ((pixel[channel] as f32 / 255.0) - mean[channel]) / std[channel],
                        );
                    }
                }
            }
        }

        let tensor = Tensor::from_shape(
            &[batch_size as usize, 3, height as usize, width as usize],
            &tensor,
        )?;
        println!("[DocOriInfer] Running inference...");
        let start = std::time::Instant::now();
        let outputs = self.base_model.run(tvec!(tensor.into()))?;
        let duration = start.elapsed();
        println!("[DocOriInfer] Inference time: {:?}", duration);
        let output_tensor = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("DocOri model did not return any outputs"))?;

        let tensor_view = output_tensor.to_array_view::<f32>()?;
        for i in 0..batch_size {
            let class_0_prob = tensor_view[[i, 0]];
            let class_270_prob = tensor_view[[i, 1]];
            let class_180_prob = tensor_view[[i, 2]];
            let class_90_prob = tensor_view[[i, 3]];
            let angle_probs = [
                (Rotation::Deg0, class_0_prob),
                (Rotation::Deg90, class_90_prob), 
                (Rotation::Deg180, class_180_prob),
                (Rotation::Deg270, class_270_prob),
            ];

            let (best_angle, best_prob) = angle_probs
                .into_iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN encountered"))
                .unwrap();

            println!("样本 {}: 最佳角度 {:?}, 概率 {:.4}", i, best_angle, best_prob);
            rotations.push(best_angle);
        }
        Ok(rotations)
    }

    pub fn run_with_timings(
        &self,
        images: Vec<DynamicImage>,
    ) -> Result<(Vec<Rotation>, StageTimings), OcrError> {
        let inference_now = Instant::now();
        let result = self
            .run(images)
            .map_err(|e| OcrError::RecognitionInference { source: e })?;
        let inference_timeing = inference_now.elapsed();
        Ok((
            result,
            StageTimings {
                preprocess: Duration::ZERO,
                inference: inference_timeing,
                postprocess: Duration::ZERO,
            },
        ))
    }
}

#[cfg(test)]
mod test {
    use crate::doc_ori::DocOriInferenceSession;

    #[test]
    fn test() {
        let s = DocOriInferenceSession::load(
            "/Users/zhangtao/Downloads/PP-LCNet_x1_0_doc_ori.onnx",
        );
        if let Ok(s) = s {
            let img_180 = image::open("/Users/zhangtao/Downloads/180degree.png").unwrap();
            let img_0 = image::open("/Users/zhangtao/Downloads/0degree.jpg").unwrap();
            let img_90 = image::open("/Users/zhangtao/Downloads/张涛学历学位/第001页.jpg").unwrap();
            print!("{:#?}", s.run(vec![img_180, img_0, img_90]).unwrap());
        }
    }
}
