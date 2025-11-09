use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{Array3, Axis};
use tract_onnx::prelude::Tensor;

/// Configuration parameters for `DetPreProcessor`.
#[derive(Debug, Clone, Copy)]
pub struct DetPreProcessorConfig {
    pub limit_side_len: u32,
}

impl Default for DetPreProcessorConfig {
    fn default() -> Self {
        Self {
            limit_side_len: 960,
        }
    }
}

/// Error returned when detection preprocessing fails.
#[derive(Debug)]
pub enum DetPreProcessorError {
    /// The provided image has zero width or height.
    EmptyImage,
}

impl std::fmt::Display for DetPreProcessorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetPreProcessorError::EmptyImage => {
                write!(f, "input image dimensions must be positive")
            }
        }
    }
}

impl std::error::Error for DetPreProcessorError {}

/// Result of detection preprocessing.
#[derive(Debug, Clone)]
pub struct PreprocessedDetInput {
    pub tensor: Tensor,
    pub resized_dims: (u32, u32),
    pub scale_ratio: f64,
}

/// DBNet detection preprocessor.
#[derive(Debug, Clone)]
pub struct DetPreProcessor {
    config: DetPreProcessorConfig,
}

impl DetPreProcessor {
    pub fn new(config: DetPreProcessorConfig) -> Self {
        Self { config }
    }

    pub fn process(
        &self,
        image: &DynamicImage,
    ) -> Result<PreprocessedDetInput, DetPreProcessorError> {
        let (orig_w, orig_h) = image.dimensions();
        if orig_w == 0 || orig_h == 0 {
            return Err(DetPreProcessorError::EmptyImage);
        }

        let (resized_w, resized_h, scale_ratio) =
            compute_resized_dims(orig_w, orig_h, self.config.limit_side_len);

        let resized = if resized_w == orig_w && resized_h == orig_h {
            image.clone()
        } else {
            image.resize(resized_w, resized_h, FilterType::Lanczos3)
        };

        let rgb_image = resized.to_rgb8();
        let array_hwc = Array3::<f32>::from_shape_fn(
            (resized_h as usize, resized_w as usize, 3),
            |(y, x, c)| rgb_image.get_pixel(x as u32, y as u32)[c] as f32 / 255.0,
        );

        let array_chw = array_hwc.permuted_axes([2, 0, 1]);
        let array_nchw = array_chw.insert_axis(Axis(0));
        let tensor: Tensor = array_nchw.into_dyn().into();

        Ok(PreprocessedDetInput {
            tensor,
            resized_dims: (resized_w, resized_h),
            scale_ratio,
        })
    }
}

fn compute_resized_dims(orig_w: u32, orig_h: u32, limit_side_len: u32) -> (u32, u32, f64) {
    if limit_side_len == 0 {
        return (orig_w, orig_h, 1.0);
    }

    let limit = limit_side_len as f64;
    let max_side = (orig_w.max(orig_h)) as f64;
    if max_side <= limit {
        return (orig_w, orig_h, 1.0);
    }

    let scale_ratio = limit / max_side;
    let resized_w = ((orig_w as f64 * scale_ratio).round().max(1.0)) as u32;
    let resized_h = ((orig_h as f64 * scale_ratio).round().max(1.0)) as u32;

    (resized_w, resized_h, scale_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn solid_image(width: u32, height: u32, value: u8) -> DynamicImage {
        let pixel = Rgb([value, value.saturating_sub(1), value.saturating_add(1)]);
        let buffer = ImageBuffer::from_pixel(width, height, pixel);
        DynamicImage::ImageRgb8(buffer)
    }

    #[test]
    fn resize_long_side_to_limit() {
        let image = solid_image(1920, 1080, 128);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig::default());

        let result = preprocessor.process(&image).unwrap();

        assert_eq!(result.resized_dims, (960, 540));
        assert!((result.scale_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn keep_original_size_when_within_limit() {
        let image = solid_image(800, 600, 64);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig::default());

        let result = preprocessor.process(&image).unwrap();

        assert_eq!(result.resized_dims, (800, 600));
        assert!((result.scale_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn tensor_shape_and_normalization() {
        let image = solid_image(320, 320, 255);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig {
            limit_side_len: 320,
        });

        let result = preprocessor.process(&image).unwrap();
        assert_eq!(result.tensor.shape(), &[1, 3, 320, 320]);

        let array = result.tensor.to_array_view::<f32>().unwrap();
        let min = array.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = array.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(min >= 0.0);
        assert!(max <= 1.0);
        assert!((max - 1.0).abs() < 1e-6);
    }
}
