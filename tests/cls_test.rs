use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use image::{imageops, GenericImageView};
use ndarray::{Array2, Axis};
use tract_onnx::prelude::*;
use tract_onnx::tract_core::anyhow::anyhow;

#[test]
fn preperare() {
    let img = image::open("/Users/zhangtao/Downloads/180degree.png")
        .map_err(|e| format!("Failed to load image: {}", e))
        .unwrap();
    let img = img.rotate180();
    img.save_with_format(
        "/Users/zhangtao/Downloads/180degree.png",
        image::ImageFormat::Png,
    )
    .unwrap();
}
#[test]
fn preprocess() {
    let img = image::open("/Users/zhangtao/Downloads/180degree.png")
        .map_err(|e| format!("Failed to load image: {}", e))
        .unwrap();
    let img = img.resize_exact(192, 48, imageops::FilterType::Triangle);
    // 获取图片尺寸
    let (w, h) = img.dimensions();
    println!("[ClsInfer] Image size: {}x{}", w, h);
    // 2. 转换为 RGB
    let rgb_img = img.to_rgb8();
    rgb_img
        .save_with_format(
            "/Users/zhangtao/Downloads/180degree-resize.png",
            image::ImageFormat::Png,
        )
        .unwrap();
}

#[test]
fn load_model() {
    let model_path = "/Users/zhangtao/Downloads/PP-LCNet_x0_25_textline_ori.onnx";
    println!("[ClsInfer] Loading cls model from {:?}", model_path);
     let img: image::DynamicImage = image::open("/Users/zhangtao/Downloads/0degree.jpg")
        .map_err(|e| format!("Failed to load image: {}", e))
        .unwrap();
    let mut inference_model = tract_onnx::onnx()
        .with_ignore_output_shapes(true)
        .model_for_path(model_path)
        .unwrap();
    // let (width, height) = img.dimensions();
    let height = 80;
    let width = 160;
    let batch = inference_model.symbol_table.sym("DynamicDimension.0");
    inference_model
        .set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![
                    batch.into(),
                    TDim::from(3),
                    TDim::from(height as i32),
                    TDim::from(width as i32)
                ],
            ),
        )
        .unwrap();

    println!("[ClsInfer] Cls model prepared");

    let runnable_model = inference_model
        .into_typed()
        .unwrap()
        .into_decluttered()
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();
   
    let img = img.resize_exact(width, height, imageops::FilterType::Triangle);
    // 2. 转换为 RGB
    let rgb_img = img.to_rgb8();
    let rgb_img_180 = img.rotate180().to_rgb8();
    // 3. 准备张量数据 [batch, channels, height, width]
    let mut tensor_data = Vec::with_capacity((2* 3 * height * width) as usize);

      // ImageNet 均值和标准差
    let mean = [0.485, 0.456, 0.406];  // R, G, B
    let std = [0.229, 0.224, 0.225];   // R, G, B

    for channel in 0..3 {  // 按通道处理：0=R, 1=G, 2=B
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb_img.get_pixel(x, y);
                tensor_data.push(((pixel[channel] as f32 / 255.0) - mean[channel])/std[channel]);
            }
        }
    }

    for channel in 0..3 {  // 按通道处理：0=R, 1=G, 2=B
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb_img_180.get_pixel(x, y);
                tensor_data.push(((pixel[channel] as f32 / 255.0) - mean[channel])/std[channel]);
            }
        }
    }

    // 4. 创建张量
    let tensor =
        Tensor::from_shape(&[2, 3, height as usize, width as usize], &tensor_data).unwrap();
    println!("[ClsInfer] Running inference...");
    let start = std::time::Instant::now();
    let outputs = runnable_model.run(tvec!(tensor.into())).unwrap();
    let duration = start.elapsed();
    println!("[ClsInfer] Inference time: {:?}", duration);
    // 6. 处理输出
    let output = &outputs[0];
    println!("[ClsInfer] Output shape: {:?}, Output is {:?}", output.shape(), output);

    if let Some(tensor_view) = output.to_array_view::<f32>().ok() {
        println!("{:?}", tensor_view);
        let batch_size = tensor_view.shape()[0];
        for batch in 0..batch_size {
            let class_0_prob = tensor_view[[batch, 0]];
            let class_1_prob = tensor_view[[batch, 1]];

            println!(
                "[ClsInfer] Batch {} - Class 0: {:.4}, Class 1: {:.4}",
                batch, class_0_prob, class_1_prob
            );

            // 判断类别
            let predicted_class = if class_0_prob > class_1_prob { 0 } else { 1 };
            let confidence = class_0_prob.max(class_1_prob);
            println!(
                "[ClsInfer] Predicted: Class {}, Confidence: {:.2}%",
                predicted_class,
                confidence * 100.0
            );
        }
    }
}
