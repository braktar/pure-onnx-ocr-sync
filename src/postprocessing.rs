use crate::detection::DetInferenceOutput;
use image::{GrayImage, Luma};
use imageproc::contours::{find_contours, Contour};
use imageproc::point::Point;
use ndarray::Array2;
use std::error::Error;
use std::fmt;

/// Configuration for `DetPostProcessor`.
#[derive(Debug, Clone, Copy)]
pub struct DetPostProcessorConfig {
    /// Probability threshold (0.0 - 1.0) applied before contour extraction.
    pub threshold: f32,
    /// Minimum contour area (in pixels) to keep.
    pub min_area: f32,
}

impl Default for DetPostProcessorConfig {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            min_area: 10.0,
        }
    }
}

/// Errors that can occur during detection post-processing.
#[derive(Debug)]
pub enum DetPostProcessorError {
    /// Probability map contained no elements.
    EmptyProbabilityMap,
    /// Failed to construct an image buffer from the probability map.
    ImageCreationFailed,
}

impl fmt::Display for DetPostProcessorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetPostProcessorError::EmptyProbabilityMap => {
                write!(f, "probability map must contain at least one element")
            }
            DetPostProcessorError::ImageCreationFailed => {
                write!(f, "failed to create grayscale image from probability map")
            }
        }
    }
}

impl Error for DetPostProcessorError {}

/// Extracts text candidate contours from the DBNet probability map.
#[derive(Debug, Clone)]
pub struct DetPostProcessor {
    config: DetPostProcessorConfig,
}

impl DetPostProcessor {
    pub fn new(config: DetPostProcessorConfig) -> Self {
        Self { config }
    }

    pub fn process(
        &self,
        output: &DetInferenceOutput,
    ) -> Result<Vec<Contour<i32>>, DetPostProcessorError> {
        self.process_probability_map(&output.probability_map)
    }

    pub fn process_probability_map(
        &self,
        probability_map: &Array2<f32>,
    ) -> Result<Vec<Contour<i32>>, DetPostProcessorError> {
        if probability_map.is_empty() {
            return Err(DetPostProcessorError::EmptyProbabilityMap);
        }

        let threshold = self.config.threshold.clamp(0.0, 1.0);

        let (height, width) = probability_map.dim();
        let mut buffer = Vec::with_capacity(height * width);

        for &value in probability_map.iter() {
            let clamped = value.clamp(0.0, 1.0);
            let byte = if clamped >= threshold { 255 } else { 0 };
            buffer.push(byte);
        }

        let mut gray = GrayImage::from_vec(width as u32, height as u32, buffer)
            .ok_or_else(|| DetPostProcessorError::ImageCreationFailed)?;

        // Ensure the binary image uses full white for foreground for consistent contour detection.
        for pixel in gray.pixels_mut() {
            *pixel = if pixel[0] > 0 { Luma([255]) } else { Luma([0]) };
        }

        let contours = find_contours::<i32>(&gray);
        let min_area = self.config.min_area.max(0.0);

        let filtered = contours
            .into_iter()
            .filter(|contour| contour.points.len() >= 3)
            .filter(|contour| contour_area(contour) >= min_area)
            .collect();

        Ok(filtered)
    }
}

fn contour_area(contour: &Contour<i32>) -> f32 {
    if contour.points.len() < 3 {
        return 0.0;
    }

    let mut area = 0f64;
    for window in contour.points.windows(2) {
        if let [Point { x: x1, y: y1 }, Point { x: x2, y: y2 }] = window {
            area += (*x1 as f64) * (*y2 as f64) - (*x2 as f64) * (*y1 as f64);
        }
    }

    let first = contour.points.first().unwrap();
    let last = contour.points.last().unwrap();
    area += (last.x as f64) * (first.y as f64) - (first.x as f64) * (last.y as f64);

    (area.abs() * 0.5) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn extracts_single_square_contour() {
        let probability_map = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];

        let processor = DetPostProcessor::new(DetPostProcessorConfig {
            threshold: 0.5,
            min_area: 1.0,
        });

        let contours = processor.process_probability_map(&probability_map).unwrap();
        assert_eq!(contours.len(), 1);

        let area = contour_area(&contours[0]);
        assert!(area >= 1.0, "expected positive area, got {}", area);
    }

    #[test]
    fn filters_small_regions() {
        let probability_map = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];

        let processor = DetPostProcessor::new(DetPostProcessorConfig {
            threshold: 0.5,
            min_area: 5.0,
        });

        let contours = processor.process_probability_map(&probability_map).unwrap();
        assert!(contours.is_empty());
    }

    #[test]
    fn empty_probability_map_is_error() {
        let probability_map = Array2::<f32>::zeros((0, 0));
        let processor = DetPostProcessor::new(DetPostProcessorConfig::default());

        let err = processor
            .process_probability_map(&probability_map)
            .unwrap_err();
        matches!(err, DetPostProcessorError::EmptyProbabilityMap);
    }
}
