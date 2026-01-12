use std::path::Path;
use std::time::{Duration, Instant};
use image::imageops::FilterType::Triangle;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::anyhow::anyhow;
use image::{DynamicImage, GenericImageView};
use crate::{OcrError, RecPreProcessorError, RecTextRegion, StageTimings};
use crate::preprocessing::Rotation;

#[derive(Debug)]
pub struct TextLineClsInferenceSession {
    base_model: TypedRunnableModel<TypedModel>,
    height: i32,
    width: i32
}

impl TextLineClsInferenceSession {
    pub fn load(model_path: impl AsRef<Path>) -> TractResult<Self> {
        let model_path = model_path.as_ref();
        println!("[TextLineClsInfer] Loading PP-LCNet_x0_25_textline_ori model from {:?}", model_path);
        let height = 80;
        let width = 160;
        let mut inference_model = tract_onnx::onnx()
            .with_ignore_output_shapes(true)
            .model_for_path(model_path)?;
        let batch_sym = inference_model.symbol_table.sym("DynamicDimension.0");
        inference_model.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![batch_sym.into(), TDim::from(3), TDim::from(height), TDim::from(width)],
            ),
        )?;
        let runnable_model = inference_model
            .into_typed()
            ?
            .into_decluttered()
            ?
            .into_optimized()
            ?
            .into_runnable()
            ?;

        println!("[TextLineClsInfer] Model PP-LCNet_x0_25_textline_ori prepared");
        Ok(Self {
            base_model: runnable_model,
            height: height,
            width: width
        })
    }

    pub fn extract_images(&self,
        image: &DynamicImage,
        regions: &[RecTextRegion],) -> Result<Vec<DynamicImage>, RecPreProcessorError> {
            if regions.is_empty() {
            return Err(RecPreProcessorError::EmptyRegions);
        }

        let (img_w, img_h) = image.dimensions();
        if img_w == 0 || img_h == 0 {
            return Err(RecPreProcessorError::EmptyImage);
        }
        let batch_size = regions.len();
        let mut crop_images = Vec::with_capacity(batch_size);
        for (index, region) in regions.iter().copied().enumerate() {
            if region.width == 0 || region.height == 0 {
                return Err(RecPreProcessorError::ZeroArea { index });
            }

            if region.x >= img_w
                || region.y >= img_h
                || region.x + region.width > img_w
                || region.y + region.height > img_h
            {
                return Err(RecPreProcessorError::RegionOutOfBounds {
                    index,
                    image_dims: (img_w, img_h),
                    region,
                });
            }
            let cropped = image.crop_imm(region.x, region.y, region.width, region.height);
            crop_images.push(cropped);
        }
        Ok(crop_images)
    }
    
    pub fn run(&self, images: Vec<DynamicImage>) -> TractResult<Vec<Rotation>> {
        let batch_size = images.len();

        println!(
            "[TextLineClsInfer] Running inference with input dims {:?}",
            batch_size
        );
        
        let (width, height) = (self.width as u32, self.height as u32);
        // ImageNet 均值和标准差
        let mean = [0.485, 0.456, 0.406];  // R, G, B
        let std = [0.229, 0.224, 0.225];   // R, G, B
        let mut tensor = Vec::with_capacity(( (batch_size as u32) * 3 * height * width) as usize);
        let mut rotations = Vec::with_capacity(batch_size);
        images.into_iter().map(|img| {
            let mut rotation = Rotation::Deg0;
            let rotated = 
            if img.height() as f32 / img.width() as f32 > 1.5 {
                rotation = Rotation::Deg90;
                img.rotate90()
            } else {
                img
            };

            let rgb_img = rotated.resize_exact(width, height, 
                Triangle).to_rgb8();
            let mut tensor_data = Vec::with_capacity((3 * height * width) as usize);
            for channel in 0..3 {
                for y in 0..height {
                    for x in 0..width {
                        let pixel = rgb_img.get_pixel(x, y);
                        tensor_data.push(((pixel[channel] as f32 / 255.0) - mean[channel])/std[channel]);
                    }
                }
            }
            (tensor_data, rotation)
        }).for_each(|(mut tensor_data, rotation)| {
            tensor.append(&mut tensor_data);
            rotations.push(rotation);
        });
        let tensor =
            Tensor::from_shape(&[batch_size as usize, 3, height as usize, width as usize], &tensor)?;
        println!("[TextLineClsInfer] Running inference...");
        let start = std::time::Instant::now();
        let outputs = self.base_model.run(tvec!(tensor.into()))?;
        let duration = start.elapsed();
        println!("[TextLineClsInfer] Inference time: {:?}", duration);
        let output_tensor = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("TextLineCls model did not return any outputs"))?;

        let tensor_view = output_tensor.to_array_view::<f32>()?;
        let batch_size = tensor_view.shape()[0];
        let mut result = Vec::with_capacity(batch_size);
        for batch in 0..batch_size {
            let mut rotation = rotations[batch];
            let class_0_prob = tensor_view[[batch, 0]];
            let class_1_prob = tensor_view[[batch, 1]];

            println!(
                "[TextLineClsInfer] Batch {} - Class 0: {:.4}, Class 1: {:.4}",
                batch, class_0_prob, class_1_prob
            );

            // 判断类别
            let predicted_class = if class_0_prob > class_1_prob { 0 } else { 1 };
            let confidence = class_0_prob.max(class_1_prob);
            println!(
                "[TextLineClsInfer] Predicted: Class {}, Confidence: {:.2}%",
                predicted_class,
                confidence * 100.0
            );
            if class_0_prob < class_1_prob {
                rotation = rotation.rotate_by(180);
            }
            result.push(rotation);
        }
        Ok(result)
    }

    
    pub fn run_with_timings(&self, image: &DynamicImage, regions: &Vec<RecTextRegion>) -> Result<(Vec<Rotation>, StageTimings), OcrError> {
        let preprocess_now = Instant::now();
        let cropeds = self.extract_images(image, regions).map_err(|e|OcrError::RecognitionPreprocess { source: e })?;
        let preprocess_timeing = preprocess_now.elapsed();
        let inference_now = Instant::now();
        let result = self.run(cropeds).map_err(|e| OcrError::RecognitionInference { source: e })?;
        let inference_timeing = inference_now.elapsed();
        Ok((result, StageTimings {
            preprocess: preprocess_timeing,
            inference: inference_timeing,
            postprocess: Duration::ZERO,
        }))
    }
}