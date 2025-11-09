use crate::detection::DetInferenceSession;
use crate::dictionary::{DictionaryError, RecDictionary};
use crate::postprocessing::DetPolygonUnclipperConfig;
use crate::preprocessing::{DetPreProcessorConfig, RecPreProcessorConfig};
use crate::recognition::{RecInferenceSession, RecPostProcessorConfig};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tract_onnx::prelude::TractError;

/// Errors that can occur while building or using the OCR engine.
#[derive(Debug)]
pub enum OcrError {
    /// A required builder field was not provided.
    MissingField { field: &'static str },
    /// An IO error occurred while accessing a resource.
    Io {
        source: std::io::Error,
        path: PathBuf,
    },
    /// Loading an ONNX model failed.
    ModelLoad { source: TractError, path: PathBuf },
    /// Loading the recognition dictionary failed.
    Dictionary { source: DictionaryError },
    /// The provided configuration contained invalid values.
    InvalidConfiguration { message: String },
}

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OcrError::MissingField { field } => {
                write!(f, "required builder field `{}` was not provided", field)
            }
            OcrError::Io { path, source } => {
                write!(f, "failed to access resource {:?}: {}", path, source)
            }
            OcrError::ModelLoad { path, source } => {
                write!(f, "failed to load ONNX model {:?}: {}", path, source)
            }
            OcrError::Dictionary { source } => write!(f, "failed to load dictionary: {}", source),
            OcrError::InvalidConfiguration { message } => write!(f, "{}", message),
        }
    }
}

impl Error for OcrError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            OcrError::MissingField { .. } => None,
            OcrError::Io { source, .. } => Some(source),
            OcrError::ModelLoad { .. } => None,
            OcrError::Dictionary { source } => Some(source),
            OcrError::InvalidConfiguration { .. } => None,
        }
    }
}

impl From<DictionaryError> for OcrError {
    fn from(source: DictionaryError) -> Self {
        Self::Dictionary { source }
    }
}

/// Aggregated configuration used by [`OcrEngine`] during inference.
#[derive(Debug, Clone)]
pub struct OcrEngineConfig {
    pub det_preprocessor: DetPreProcessorConfig,
    pub det_unclipper: DetPolygonUnclipperConfig,
    pub rec_preprocessor: RecPreProcessorConfig,
    pub rec_postprocessor: RecPostProcessorConfig,
    pub rec_batch_size: usize,
}

impl Default for OcrEngineConfig {
    fn default() -> Self {
        Self {
            det_preprocessor: DetPreProcessorConfig::default(),
            det_unclipper: DetPolygonUnclipperConfig::default(),
            rec_preprocessor: RecPreProcessorConfig::default(),
            rec_postprocessor: RecPostProcessorConfig::default(),
            rec_batch_size: 8,
        }
    }
}

/// Fully prepared OCR engine with loaded models and assets.
#[derive(Debug)]
pub struct OcrEngine {
    pub(crate) det_model_path: PathBuf,
    pub(crate) rec_model_path: PathBuf,
    pub(crate) dictionary_path: PathBuf,
    pub(crate) det_session: Arc<DetInferenceSession>,
    pub(crate) rec_session: Arc<RecInferenceSession>,
    pub(crate) dictionary: Arc<RecDictionary>,
    pub(crate) config: OcrEngineConfig,
}

impl OcrEngine {
    fn new(
        det_model_path: PathBuf,
        rec_model_path: PathBuf,
        dictionary_path: PathBuf,
        det_session: DetInferenceSession,
        rec_session: RecInferenceSession,
        dictionary: RecDictionary,
        config: OcrEngineConfig,
    ) -> Self {
        Self {
            det_model_path,
            rec_model_path,
            dictionary_path,
            det_session: Arc::new(det_session),
            rec_session: Arc::new(rec_session),
            dictionary: Arc::new(dictionary),
            config,
        }
    }

    /// Returns the effective configuration for this engine.
    pub fn config(&self) -> &OcrEngineConfig {
        &self.config
    }
}

/// Builder for constructing [`OcrEngine`] instances.
#[derive(Debug, Clone)]
pub struct OcrEngineBuilder {
    det_model_path: Option<PathBuf>,
    rec_model_path: Option<PathBuf>,
    dictionary_path: Option<PathBuf>,
    det_limit_side_len: u32,
    det_unclip_ratio: f32,
    rec_batch_size: usize,
}

impl Default for OcrEngineBuilder {
    fn default() -> Self {
        Self {
            det_model_path: None,
            rec_model_path: None,
            dictionary_path: None,
            det_limit_side_len: DetPreProcessorConfig::default().limit_side_len,
            det_unclip_ratio: DetPolygonUnclipperConfig::default().unclip_ratio,
            rec_batch_size: OcrEngineConfig::default().rec_batch_size,
        }
    }
}

impl OcrEngineBuilder {
    /// Creates a new builder instance using default configuration values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the path to the DBNet detection ONNX model.
    pub fn det_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.det_model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the path to the SVTR recognition ONNX model.
    pub fn rec_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.rec_model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the path to the recognition dictionary file.
    pub fn dictionary_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.dictionary_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the maximum side length for detection preprocessing.
    pub fn det_limit_side_len(mut self, len: u32) -> Self {
        self.det_limit_side_len = len;
        self
    }

    /// Sets the unclip ratio used during polygon offsetting.
    pub fn det_unclip_ratio(mut self, ratio: f64) -> Self {
        self.det_unclip_ratio = ratio as f32;
        self
    }

    /// Sets the maximum batch size for recognition.
    pub fn rec_batch_size(mut self, size: usize) -> Self {
        self.rec_batch_size = size;
        self
    }

    /// Consumes the builder and attempts to construct an [`OcrEngine`].
    pub fn build(self) -> Result<OcrEngine, OcrError> {
        let det_model_path = self.det_model_path.ok_or(OcrError::MissingField {
            field: "det_model_path",
        })?;
        let rec_model_path = self.rec_model_path.ok_or(OcrError::MissingField {
            field: "rec_model_path",
        })?;
        let dictionary_path = self.dictionary_path.ok_or(OcrError::MissingField {
            field: "dictionary_path",
        })?;

        if self.rec_batch_size == 0 {
            return Err(OcrError::InvalidConfiguration {
                message: "rec_batch_size must be greater than zero".to_string(),
            });
        }

        verify_file_exists(&det_model_path)?;
        verify_file_exists(&rec_model_path)?;
        verify_file_exists(&dictionary_path)?;

        let det_session =
            DetInferenceSession::load(&det_model_path).map_err(|source| OcrError::ModelLoad {
                source,
                path: det_model_path.clone(),
            })?;
        let rec_session =
            RecInferenceSession::load(&rec_model_path).map_err(|source| OcrError::ModelLoad {
                source,
                path: rec_model_path.clone(),
            })?;
        let dictionary = RecDictionary::from_path(&dictionary_path)?;

        let mut det_unclipper_config = DetPolygonUnclipperConfig::default();
        det_unclipper_config.unclip_ratio = self.det_unclip_ratio;

        let mut det_preprocessor_config = DetPreProcessorConfig::default();
        det_preprocessor_config.limit_side_len = self.det_limit_side_len;

        let config = OcrEngineConfig {
            det_preprocessor: det_preprocessor_config,
            det_unclipper: det_unclipper_config,
            rec_preprocessor: RecPreProcessorConfig::default(),
            rec_postprocessor: RecPostProcessorConfig::default(),
            rec_batch_size: self.rec_batch_size,
        };

        Ok(OcrEngine::new(
            det_model_path,
            rec_model_path,
            dictionary_path,
            det_session,
            rec_session,
            dictionary,
            config,
        ))
    }
}

fn verify_file_exists(path: &Path) -> Result<(), OcrError> {
    if let Err(source) = fs::metadata(path) {
        return Err(OcrError::Io {
            source,
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn existing_model_paths() -> Option<(PathBuf, PathBuf, PathBuf)> {
        let det = Path::new("models/ppocrv5/det.onnx");
        let rec = Path::new("models/ppocrv5/rec.onnx");
        let dict = Path::new("models/ppocrv5/ppocrv5_dict.txt");
        if det.exists() && rec.exists() && dict.exists() {
            Some((det.to_path_buf(), rec.to_path_buf(), dict.to_path_buf()))
        } else {
            None
        }
    }

    #[test]
    fn missing_det_model_path_returns_error() {
        let err = OcrEngineBuilder::new()
            .rec_model_path("rec.onnx")
            .dictionary_path("dict.txt")
            .build()
            .unwrap_err();

        match err {
            OcrError::MissingField { field } => assert_eq!(field, "det_model_path"),
            other => panic!("expected MissingField error, got {:?}", other),
        }
    }

    #[test]
    fn missing_dictionary_path_returns_error() {
        let err = OcrEngineBuilder::new()
            .det_model_path("det.onnx")
            .rec_model_path("rec.onnx")
            .build()
            .unwrap_err();

        match err {
            OcrError::MissingField { field } => assert_eq!(field, "dictionary_path"),
            other => panic!("expected MissingField error, got {:?}", other),
        }
    }

    #[test]
    fn zero_recognition_batch_size_is_rejected() {
        let err = OcrEngineBuilder::new()
            .det_model_path("det.onnx")
            .rec_model_path("rec.onnx")
            .dictionary_path("dict.txt")
            .rec_batch_size(0)
            .build()
            .unwrap_err();

        match err {
            OcrError::InvalidConfiguration { message } => {
                assert!(message.contains("rec_batch_size"));
            }
            other => panic!("expected InvalidConfiguration error, got {:?}", other),
        }
    }

    #[test]
    fn build_succeeds_when_paths_exist() {
        let (det, rec, dict) = existing_model_paths()
            .expect("expected PP-OCRv5 assets to be present under models/ppocrv5/");

        let engine = OcrEngineBuilder::new()
            .det_model_path(&det)
            .rec_model_path(&rec)
            .dictionary_path(&dict)
            .det_limit_side_len(1024)
            .det_unclip_ratio(2.0)
            .rec_batch_size(4)
            .build()
            .expect("engine should build successfully");

        assert_eq!(engine.config().det_preprocessor.limit_side_len, 1024);
        assert!((engine.config().det_unclipper.unclip_ratio - 2.0).abs() < f32::EPSILON);
        assert_eq!(engine.config().rec_batch_size, 4);
    }
}
