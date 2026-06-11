use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{s, Array3, Array4, Axis};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rotation {
    Deg0 = 0,
    Deg90 = 90,
    Deg180 = 180,
    Deg270 = 270,
}

impl Rotation {
    pub fn from_degrees(degrees: i32) -> Self {
        match degrees.rem_euclid(360) {
            0 => Rotation::Deg0,
            90 | -270 => Rotation::Deg90,
            180 | -180 => Rotation::Deg180,
            270 | -90 => Rotation::Deg270,
            _ => unreachable!(), // 规范化后只可能是这四个值
        }
    }
    pub fn degrees(self) -> i32 {
        self as i32
    }

    pub fn shortest_rotation_to_zero(&self) -> i32 {
        match self {
            Rotation::Deg0 => 0,
            Rotation::Deg90 => 270,
            Rotation::Deg180 => 180,
            Rotation::Deg270 => 90,
        }
    }
    pub fn rotate_by(self, degrees: i32) -> Self {
        Self::from_degrees(self.degrees() + degrees)
    }
}

impl From<i32> for Rotation {
    fn from(degrees: i32) -> Self {
        Self::from_degrees(degrees)
    }
}

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

    pub fn limit_side_len(&self) -> u32 {
        self.config.limit_side_len
    }

    pub fn process(
        &self,
        image: &DynamicImage,
    ) -> Result<PreprocessedDetInput, DetPreProcessorError> {
        self.process_with_limit(image, self.config.limit_side_len)
    }

    /// Runs detection preprocessing with an explicit long-side limit (multi-scale retry).
    pub fn process_with_limit(
        &self,
        image: &DynamicImage,
        limit_side_len: u32,
    ) -> Result<PreprocessedDetInput, DetPreProcessorError> {
        let (orig_w, orig_h) = image.dimensions();
        if orig_w == 0 || orig_h == 0 {
            return Err(DetPreProcessorError::EmptyImage);
        }

        let (_, _, scale_ratio) = compute_resized_dims(orig_w, orig_h, limit_side_len);
        let (tensor_w, tensor_h) =
            detection_tensor_dims(orig_w, orig_h, limit_side_len);

        let resized = image.resize_exact(tensor_w, tensor_h, FilterType::Lanczos3);
        let rgb_image = resized.to_rgb8();

        let mut array_hwc = Array3::<f32>::zeros((tensor_h as usize, tensor_w as usize, 3));

        for y in 0..tensor_h as usize {
            for x in 0..tensor_w as usize {
                let pixel = rgb_image.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    array_hwc[[y, x, c]] = pixel[c] as f32 / 255.0;
                }
            }
        }

        let array_chw = array_hwc.permuted_axes([2, 0, 1]);
        let array_nchw = array_chw.insert_axis(Axis(0));
        let tensor: Tensor = array_nchw.into_dyn().into();

        Ok(PreprocessedDetInput {
            tensor,
            resized_dims: (tensor_w, tensor_h),
            scale_ratio,
        })
    }
}

/// DBNet tensor width/height snapped to multiples of 32 (tract-safe, no zero padding).
pub fn detection_tensor_dims(orig_w: u32, orig_h: u32, limit_side_len: u32) -> (u32, u32) {
    let (resized_w, resized_h, _) = compute_resized_dims(orig_w, orig_h, limit_side_len);
    (
        round_up_to_multiple(resized_w, 32).max(32),
        round_up_to_multiple(resized_h, 32).max(32),
    )
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

fn round_up_to_multiple(value: u32, multiple: u32) -> u32 {
    if multiple == 0 {
        return value;
    }

    let remainder = value % multiple;
    if remainder == 0 {
        value
    } else {
        value + multiple - remainder
    }
}

/// Snap recognition tensor width to tract-safe values (multiples of 32).
pub fn snap_recognition_width(width: u32, max_width: u32) -> u32 {
    let snapped = round_up_to_multiple(width.max(32), 32).min(max_width);
    snapped.max(32)
}

/// Rectangle specifying the area to crop for recognition preprocessing.
#[derive(Debug, Clone, Copy)]
pub struct RecTextRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Configuration parameters for recognition preprocessing.
#[derive(Debug, Clone)]
pub struct RecPreProcessorConfig {
    pub target_height: u32,
    pub max_width: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub pad_value: [f32; 3],
}

impl Default for RecPreProcessorConfig {
    fn default() -> Self {
        Self {
            target_height: 48,
            max_width: 320,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            pad_value: [0.0, 0.0, 0.0],
        }
    }
}

/// Errors that can be produced by recognition preprocessing.
#[derive(Debug)]
pub enum RecPreProcessorError {
    /// The provided batch of regions is empty.
    EmptyRegions,
    /// The input image has zero width or height.
    EmptyImage,
    /// The configuration contains an invalid parameter (e.g. zero height/width).
    InvalidConfiguration,
    /// A region had zero width or height.
    ZeroArea { index: usize },
    /// A region extended beyond the bounds of the image.
    RegionOutOfBounds {
        index: usize,
        image_dims: (u32, u32),
        region: RecTextRegion,
    },
}

impl std::fmt::Display for RecPreProcessorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecPreProcessorError::EmptyRegions => {
                write!(f, "at least one text region is required for recognition")
            }
            RecPreProcessorError::EmptyImage => {
                write!(f, "input image dimensions must be positive")
            }
            RecPreProcessorError::InvalidConfiguration => {
                write!(f, "recognition preprocessor configuration is invalid")
            }
            RecPreProcessorError::ZeroArea { index } => {
                write!(f, "text region at index {} has zero area", index)
            }
            RecPreProcessorError::RegionOutOfBounds {
                index,
                image_dims,
                region,
            } => write!(
                f,
                "text region at index {} (x={}, y={}, w={}, h={}) exceeds image bounds {:?}",
                index, region.x, region.y, region.width, region.height, image_dims
            ),
        }
    }
}

impl std::error::Error for RecPreProcessorError {}

/// Result of recognition preprocessing.
#[derive(Debug, Clone)]
pub struct PreprocessedRecBatch {
    pub tensor: Tensor,
    pub valid_widths: Vec<u32>,
    pub max_width: u32,
}

impl PreprocessedRecBatch {
    pub fn valid_width_ratios(&self) -> Vec<f32> {
        if self.max_width == 0 {
            return vec![0.0; self.valid_widths.len()];
        }
        self.valid_widths
            .iter()
            .map(|width| *width as f32 / self.max_width as f32)
            .collect()
    }
}

/// SVTR recognition preprocessor.
#[derive(Debug, Clone)]
pub struct RecPreProcessor {
    config: RecPreProcessorConfig,
}

impl RecPreProcessor {
    pub fn new(config: RecPreProcessorConfig) -> Self {
        Self { config }
    }
    pub fn process(
        &self,
        image: &DynamicImage,
        regions: &[RecTextRegion],
        rotations: &[Rotation],
    ) -> Result<PreprocessedRecBatch, RecPreProcessorError> {
        if regions.is_empty() {
            return Err(RecPreProcessorError::EmptyRegions);
        }

        if self.config.target_height == 0 || self.config.max_width == 0 {
            return Err(RecPreProcessorError::InvalidConfiguration);
        }

        let (img_w, img_h) = image.dimensions();
        if img_w == 0 || img_h == 0 {
            return Err(RecPreProcessorError::EmptyImage);
        }

        let target_height = self.config.target_height;
        let max_width = self.config.max_width;
        let batch_size = regions.len();

        let mut batch =
            Array4::<f32>::zeros((batch_size, 3, target_height as usize, max_width as usize));

        for sample in 0..batch_size {
            for channel in 0..3 {
                let pad = normalize_value(
                    self.config.pad_value[channel],
                    self.config.mean[channel],
                    self.config.std[channel],
                );
                batch.slice_mut(s![sample, channel, .., ..]).fill(pad);
            }
        }

        let mut valid_widths = Vec::with_capacity(batch_size);

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
            let rotation = if index < rotations.len() {
                rotations[index]
            } else {
                Rotation::Deg0
            };

            let mut cropped = image.crop_imm(region.x, region.y, region.width, region.height);
            cropped = match rotation {
                Rotation::Deg0 => cropped,
                Rotation::Deg90 => cropped.rotate270(),
                Rotation::Deg180 => cropped.rotate180(),
                Rotation::Deg270 => cropped.rotate90(),
            };
            let aspect_ratio = cropped.width() as f32 / cropped.height() as f32;
            let mut target_width = snap_recognition_width(
                (aspect_ratio * target_height as f32).round().max(1.0) as u32,
                max_width,
            );
            if target_width == 0 {
                target_width = 32;
            }

            let resized = cropped.resize_exact(target_width, target_height, FilterType::Lanczos3);
            let rgb_image = resized.to_rgb8();

            for y in 0..target_height as usize {
                for x in 0..target_width as usize {
                    let pixel = rgb_image.get_pixel(x as u32, y as u32);
                    for channel in 0..3 {
                        let value = pixel[channel] as f32 / 255.0;
                        let normalized = normalize_value(
                            value,
                            self.config.mean[channel],
                            self.config.std[channel],
                        );
                        batch[[index, channel, y, x]] = normalized;
                    }
                }
            }

            valid_widths.push(target_width);
        }

        let tensor: Tensor = batch.into_dyn().into();
        Ok(PreprocessedRecBatch {
            tensor,
            valid_widths,
            max_width,
        })
    }
}

fn normalize_value(value: f32, mean: f32, std: f32) -> f32 {
    if std == 0.0 {
        0.0
    } else {
        (value - mean) / std
    }
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

    fn gradient_image(width: u32, height: u32) -> DynamicImage {
        let mut buffer = ImageBuffer::new(width, height);
        for (x, y, pixel) in buffer.enumerate_pixels_mut() {
            let base = ((x + y) % 256) as u8;
            let green = base.saturating_add(32);
            let blue = base.saturating_add(64);
            *pixel = Rgb([base, green, blue]);
        }
        DynamicImage::ImageRgb8(buffer)
    }

    #[test]
    fn resize_long_side_to_limit() {
        let image = solid_image(1920, 1080, 128);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig::default());

        let result = preprocessor.process(&image).unwrap();

        assert_eq!(result.resized_dims, (960, 544));
        assert!((result.scale_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn keep_original_size_when_within_limit() {
        let image = solid_image(800, 600, 64);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig::default());

        let result = preprocessor.process(&image).unwrap();

        assert_eq!(result.resized_dims, (800, 608));
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

    #[test]
    fn detection_tensor_dims_are_padded_to_multiple_of_32() {
        let image = solid_image(123, 77, 200);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig::default());

        let result = preprocessor.process(&image).unwrap();

        assert_eq!(result.resized_dims, (128, 96));
        assert_eq!(result.tensor.shape(), &[1, 3, 96, 128]);
        assert_eq!(detection_tensor_dims(123, 77, 960), (128, 96));
        assert!((result.scale_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn recognition_single_region_preprocessing() {
        let image = gradient_image(200, 100);
        let config = RecPreProcessorConfig::default();
        let regions = vec![RecTextRegion {
            x: 20,
            y: 10,
            width: 80,
            height: 40,
        }];
        let rotations = vec![Rotation::Deg0];
        let preprocessor = RecPreProcessor::new(config.clone());
        let batch = preprocessor.process(&image, &regions, &rotations).unwrap();

        let expected_shape = [
            1,
            3,
            config.target_height as usize,
            config.max_width as usize,
        ];
        assert_eq!(batch.tensor.shape(), &expected_shape);
        assert_eq!(batch.valid_widths, vec![96]);

        let tensor = batch.tensor.to_array_view::<f32>().unwrap();
        let pad = normalize_value(config.pad_value[0], config.mean[0], config.std[0]);
        assert!(
            (tensor[[0, 0, 0, (config.max_width - 1) as usize]] - pad).abs() < 1e-6,
            "padded area should remain at pad value"
        );
        assert!(
            (tensor[[0, 0, 0, 0]] - pad).abs() > 1e-3,
            "cropped content should differ from pad value"
        );

        let ratios = batch.valid_width_ratios();
        assert_eq!(ratios.len(), 1);
        assert!((ratios[0] - 96.0 / config.max_width as f32).abs() < f32::EPSILON);
    }

    #[test]
    fn recognition_multiple_regions_padding() {
        let image = gradient_image(320, 160);
        let config = RecPreProcessorConfig::default();
        let regions = vec![
            RecTextRegion {
                x: 0,
                y: 0,
                width: 120,
                height: 60,
            },
            RecTextRegion {
                x: 150,
                y: 40,
                width: 40,
                height: 80,
            },
        ];

        let preprocessor = RecPreProcessor::new(config.clone());
        let rotations = vec![Rotation::Deg0, Rotation::Deg0];
        let batch = preprocessor.process(&image, &regions, &rotations).unwrap();

        assert_eq!(batch.valid_widths, vec![96, 24]);

        let tensor = batch.tensor.to_array_view::<f32>().unwrap();
        let pad = normalize_value(config.pad_value[0], config.mean[0], config.std[0]);

        // Ensure padding column for first sample is untouched.
        assert!((tensor[[0, 0, 10, (config.max_width - 1) as usize]] - pad).abs() < 1e-6);
        // Ensure padding column for second sample is untouched.
        assert!((tensor[[1, 1, 20, (config.max_width - 1) as usize]] - pad).abs() < 1e-6);
    }

    #[test]
    fn recognition_region_out_of_bounds_is_error() {
        let image = gradient_image(100, 50);
        let config = RecPreProcessorConfig::default();
        let regions = vec![RecTextRegion {
            x: 80,
            y: 10,
            width: 30,
            height: 20,
        }];

        let rotations = vec![Rotation::Deg0];
        let preprocessor = RecPreProcessor::new(config);
        let error = preprocessor
            .process(&image, &regions, &rotations)
            .unwrap_err();
        assert!(matches!(
            error,
            RecPreProcessorError::RegionOutOfBounds { index: 0, .. }
        ));
    }

    #[test]
    fn recognition_zero_area_region_is_error() {
        let image = gradient_image(100, 50);
        let config = RecPreProcessorConfig::default();
        let regions = vec![RecTextRegion {
            x: 10,
            y: 10,
            width: 0,
            height: 20,
        }];

        let rotations = vec![Rotation::Deg0];
        let preprocessor = RecPreProcessor::new(config);
        let error = preprocessor.process(&image, &regions, &rotations).unwrap_err();
        assert!(matches!(error, RecPreProcessorError::ZeroArea { index: 0 }));
    }
}
