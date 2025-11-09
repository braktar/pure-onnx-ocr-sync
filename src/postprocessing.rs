use crate::detection::DetInferenceOutput;
use geo_types::{Coord, LineString, Polygon};
use i_overlay::float::overlay::OverlayOptions;
use i_overlay::mesh::outline::offset::OutlineOffset;
use i_overlay::mesh::style::{LineJoin, OutlineStyle};
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
            .ok_or(DetPostProcessorError::ImageCreationFailed)?;

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

/// Corner join style for unclip offsetting.
#[derive(Debug, Clone, Copy)]
pub enum DetUnclipLineJoin {
    Bevel,
    Miter(f32),
    Round(f32),
}

impl DetUnclipLineJoin {
    fn to_line_join(self) -> LineJoin<f64> {
        match self {
            DetUnclipLineJoin::Bevel => LineJoin::Bevel,
            DetUnclipLineJoin::Miter(angle) => LineJoin::Miter(angle.max(0.01) as f64),
            DetUnclipLineJoin::Round(angle) => LineJoin::Round(angle.max(0.01) as f64),
        }
    }
}

/// Configuration for polygon offsetting (unclip).
#[derive(Debug, Clone, Copy)]
pub struct DetPolygonUnclipperConfig {
    /// Ratio applied to the DBNet area/perimeter heuristic.
    pub unclip_ratio: f32,
    /// Additional minimum area after unclipping; polygons smaller than this are discarded.
    pub min_result_area: f32,
    /// Join style applied to buffered corners.
    pub join_style: DetUnclipLineJoin,
}

impl Default for DetPolygonUnclipperConfig {
    fn default() -> Self {
        Self {
            unclip_ratio: 1.5,
            min_result_area: 25.0,
            join_style: DetUnclipLineJoin::Round(0.1),
        }
    }
}

/// Applies DBNet-style polygon offsetting (unclip) using `i_overlay`.
#[derive(Debug, Clone)]
pub struct DetPolygonUnclipper {
    config: DetPolygonUnclipperConfig,
}

impl DetPolygonUnclipper {
    pub fn new(config: DetPolygonUnclipperConfig) -> Self {
        Self { config }
    }

    pub fn unclip_contours(&self, contours: &[Contour<i32>]) -> Vec<Polygon<f64>> {
        contours
            .iter()
            .filter_map(|contour| contour_to_polygon(contour))
            .flat_map(|polygon| self.unclip_polygon(&polygon))
            .filter(|polygon| polygon_area(polygon) >= self.config.min_result_area)
            .collect()
    }

    fn unclip_polygon(&self, polygon: &Polygon<f64>) -> Vec<Polygon<f64>> {
        let distance = unclip_distance(polygon, self.config.unclip_ratio.max(0.0));
        if distance <= f64::EPSILON {
            return vec![polygon.clone()];
        }

        let shape = polygon_to_shape(polygon);
        let style = OutlineStyle::default()
            .outer_offset(distance)
            .inner_offset(0.0)
            .line_join(self.config.join_style.to_line_join());

        let options = OverlayOptions::default();
        shape
            .outline_custom(&style, options)
            .into_iter()
            .filter_map(shape_to_polygon)
            .collect()
    }
}

fn contour_to_polygon(contour: &Contour<i32>) -> Option<Polygon<f64>> {
    if contour.points.len() < 3 {
        return None;
    }

    let mut coords: Vec<Coord<f64>> = contour
        .points
        .iter()
        .map(|point| Coord {
            x: point.x as f64,
            y: point.y as f64,
        })
        .collect();

    close_if_needed(&mut coords);

    // Ensure outer contour is counter-clockwise.
    if signed_area_coords(&coords) < 0.0 {
        coords.reverse();
        close_if_needed(&mut coords);
    }

    let exterior = LineString::from(coords);
    Some(Polygon::new(exterior, Vec::new()))
}

fn polygon_to_shape(polygon: &Polygon<f64>) -> Vec<Vec<[f64; 2]>> {
    let mut shape = Vec::with_capacity(1 + polygon.interiors().len());
    shape.push(linestring_to_points(polygon.exterior(), true));
    for interior in polygon.interiors() {
        shape.push(linestring_to_points(interior, false));
    }
    shape
}

fn shape_to_polygon(shape: Vec<Vec<[f64; 2]>>) -> Option<Polygon<f64>> {
    if shape.is_empty() {
        return None;
    }

    let exterior = LineString::from(points_to_coords(&shape[0]));
    let interiors = shape
        .iter()
        .skip(1)
        .map(|points| LineString::from(points_to_coords(points)))
        .collect();

    Some(Polygon::new(exterior, interiors))
}

fn linestring_to_points(line: &LineString<f64>, want_ccw: bool) -> Vec<[f64; 2]> {
    let mut coords: Vec<Coord<f64>> = line
        .points()
        .map(|p| Coord { x: p.x(), y: p.y() })
        .collect();
    close_if_needed(&mut coords);

    let area = signed_area_coords(&coords);
    if want_ccw && area < 0.0 || !want_ccw && area > 0.0 {
        coords.reverse();
        close_if_needed(&mut coords);
    }

    coords.iter().map(|c| [c.x, c.y]).collect()
}

fn points_to_coords(points: &[[f64; 2]]) -> Vec<Coord<f64>> {
    let mut coords: Vec<Coord<f64>> = points
        .iter()
        .map(|point| Coord {
            x: point[0],
            y: point[1],
        })
        .collect();
    close_if_needed(&mut coords);
    coords
}

fn close_if_needed(coords: &mut Vec<Coord<f64>>) {
    if coords.len() < 2 {
        return;
    }
    let first = coords.first().copied().unwrap();
    let last = coords.last().copied().unwrap();
    if first.x != last.x || first.y != last.y {
        coords.push(first);
    }
}

fn signed_area_coords(coords: &[Coord<f64>]) -> f64 {
    if coords.len() < 2 {
        return 0.0;
    }

    let mut area = 0.0;
    for window in coords.windows(2) {
        if let [a, b] = window {
            area += a.x * b.y - b.x * a.y;
        }
    }
    area * 0.5
}

fn perimeter_coords(coords: &[Coord<f64>]) -> f64 {
    if coords.len() < 2 {
        return 0.0;
    }

    let mut length = 0.0;
    for window in coords.windows(2) {
        if let [a, b] = window {
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            length += (dx * dx + dy * dy).sqrt();
        }
    }
    length
}

fn polygon_area(polygon: &Polygon<f64>) -> f32 {
    let mut area = signed_area_coords(&points_to_coords(&linestring_to_points(
        polygon.exterior(),
        true,
    )))
    .abs();

    for interior in polygon.interiors() {
        area -= signed_area_coords(&points_to_coords(&linestring_to_points(interior, false))).abs();
    }

    area as f32
}

fn unclip_distance(polygon: &Polygon<f64>, ratio: f32) -> f64 {
    if ratio <= 0.0 {
        return 0.0;
    }

    let exterior_coords = points_to_coords(&linestring_to_points(polygon.exterior(), true));
    let area = signed_area_coords(&exterior_coords).abs();
    let perimeter = perimeter_coords(&exterior_coords);

    if perimeter <= f64::EPSILON {
        0.0
    } else {
        (area / perimeter) * ratio as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use imageproc::contours::BorderType;
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

    #[test]
    fn unclip_makes_polygon_larger() {
        let contour = Contour::new(
            vec![
                Point::new(0, 0),
                Point::new(4, 0),
                Point::new(4, 4),
                Point::new(0, 4),
            ],
            BorderType::Outer,
            None,
        );

        let unclipper = DetPolygonUnclipper::new(DetPolygonUnclipperConfig {
            unclip_ratio: 2.0,
            min_result_area: 1.0,
            join_style: DetUnclipLineJoin::Round(0.1),
        });

        let unclipped = unclipper.unclip_contours(&[contour]);
        assert!(!unclipped.is_empty());

        let original_area = 16.0;
        let enlarged = unclipped
            .iter()
            .map(|poly| polygon_area(poly) as f64)
            .fold(0.0, f64::max);

        assert!(
            enlarged > original_area,
            "expected unclip area ({}) to exceed original ({})",
            enlarged,
            original_area
        );
    }
}
