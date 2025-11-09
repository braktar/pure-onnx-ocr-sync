use crate::dictionary::RecDictionary;
use ndarray::{s, Array3, Axis};

/// Configuration options for greedy CTC decoding.
#[derive(Debug, Clone, Copy)]
pub struct CtcGreedyDecoderConfig {
    pub blank_id: usize,
}

impl Default for CtcGreedyDecoderConfig {
    fn default() -> Self {
        Self { blank_id: 0 }
    }
}

/// Errors that can be produced during greedy CTC decoding.
#[derive(Debug)]
pub enum CtcGreedyDecoderError {
    /// The specified blank id is not representable within the class dimension.
    BlankIdOutOfRange { blank_id: usize, class_count: usize },
    /// The provided `valid_timesteps` slice length does not match the batch size.
    TimestepsLengthMismatch { expected: usize, actual: usize },
    /// The decoder encountered an index that is not present in the dictionary.
    DictionaryIndexMissing {
        index: usize,
        dictionary_size: usize,
    },
    /// The decoder was asked to process an empty batch.
    EmptyBatch,
}

impl std::fmt::Display for CtcGreedyDecoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CtcGreedyDecoderError::BlankIdOutOfRange {
                blank_id,
                class_count,
            } => write!(
                f,
                "blank id {} is out of range for class dimension size {}",
                blank_id, class_count
            ),
            CtcGreedyDecoderError::TimestepsLengthMismatch { expected, actual } => write!(
                f,
                "valid_timesteps length {} does not match batch size {}",
                actual, expected
            ),
            CtcGreedyDecoderError::DictionaryIndexMissing {
                index,
                dictionary_size,
            } => write!(
                f,
                "decoded index {} is not present in dictionary (size {})",
                index, dictionary_size
            ),
            CtcGreedyDecoderError::EmptyBatch => {
                write!(f, "ctc decoder received an empty batch")
            }
        }
    }
}

impl std::error::Error for CtcGreedyDecoderError {}

/// Result of decoding a single sequence.
#[derive(Debug, Clone)]
pub struct DecodedSequence {
    pub text: String,
    pub token_indices: Vec<usize>,
    pub confidence: f32,
}

/// Greedy CTC decoder producing text using the recognition dictionary.
#[derive(Debug, Clone)]
pub struct CtcGreedyDecoder {
    config: CtcGreedyDecoderConfig,
}

impl CtcGreedyDecoder {
    pub fn new(config: CtcGreedyDecoderConfig) -> Self {
        Self { config }
    }

    pub fn decode(
        &self,
        logits: &Array3<f32>,
        valid_timesteps: &[usize],
        dictionary: &RecDictionary,
    ) -> Result<Vec<DecodedSequence>, CtcGreedyDecoderError> {
        let batch_size = logits.len_of(Axis(0));
        let time_steps = logits.len_of(Axis(1));
        let class_count = logits.len_of(Axis(2));

        if batch_size == 0 {
            return Err(CtcGreedyDecoderError::EmptyBatch);
        }

        if self.config.blank_id >= class_count {
            return Err(CtcGreedyDecoderError::BlankIdOutOfRange {
                blank_id: self.config.blank_id,
                class_count,
            });
        }

        if valid_timesteps.len() != batch_size {
            return Err(CtcGreedyDecoderError::TimestepsLengthMismatch {
                expected: batch_size,
                actual: valid_timesteps.len(),
            });
        }

        let mut results = Vec::with_capacity(batch_size);
        for batch_index in 0..batch_size {
            let max_steps = valid_timesteps[batch_index].min(time_steps);
            let mut previous_symbol: Option<usize> = None;
            let mut text = String::new();
            let mut token_indices = Vec::new();
            let mut confidence_sum = 0.0f32;
            let mut confidence_count = 0usize;

            for t in 0..max_steps {
                let step = logits.slice(s![batch_index, t, ..]);

                let (mut best_index, mut best_value) = (0usize, f32::NEG_INFINITY);
                for (idx, value) in step.iter().enumerate() {
                    if value > &best_value {
                        best_value = *value;
                        best_index = idx;
                    }
                }

                if best_index == self.config.blank_id {
                    previous_symbol = None;
                    continue;
                }

                if Some(best_index) == previous_symbol {
                    continue;
                }

                if best_index >= dictionary.len() {
                    return Err(CtcGreedyDecoderError::DictionaryIndexMissing {
                        index: best_index,
                        dictionary_size: dictionary.len(),
                    });
                }

                let token = dictionary
                    .token(best_index)
                    .expect("dictionary bounds verified above");
                text.push_str(token);
                token_indices.push(best_index);

                let mut sum = 0.0f32;
                for value in step.iter() {
                    sum += (value - best_value).exp();
                }
                let probability = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                confidence_sum += probability;
                confidence_count += 1;

                previous_symbol = Some(best_index);
            }

            let confidence = if confidence_count == 0 {
                0.0
            } else {
                confidence_sum / confidence_count as f32
            };

            results.push(DecodedSequence {
                text,
                token_indices,
                confidence,
            });
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn dictionary_from_tokens(tokens: &[&str]) -> RecDictionary {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ctc_dict_{}.txt", timestamp));
        fs::write(&path, tokens.join("\n")).unwrap();
        let dict = RecDictionary::from_path(&path).unwrap();
        fs::remove_file(path).ok();
        dict
    }

    #[test]
    fn decodes_sequence_with_duplicates_and_blank() {
        let logits = Array3::from_shape_vec(
            (1, 4, 3),
            vec![
                2.0, 0.5, -1.0, //
                2.5, 0.4, -2.0, //
                -0.2, -0.3, 3.0, //
                0.1, 2.2, -0.5, //
            ],
        )
        .unwrap();

        let decoder = CtcGreedyDecoder::new(CtcGreedyDecoderConfig { blank_id: 2 });
        let dictionary = dictionary_from_tokens(&["a", "b"]);
        let sequences = decoder
            .decode(&logits, &[4], &dictionary)
            .expect("decoding should succeed");

        assert_eq!(sequences.len(), 1);
        let first = &sequences[0];
        assert_eq!(first.text, "ab");
        assert_eq!(first.token_indices, vec![0, 1]);
        assert!(first.confidence > 0.0);
        assert!(first.confidence <= 1.0);
    }

    #[test]
    fn handles_all_blank_sequence() {
        let logits = Array3::from_shape_vec(
            (1, 3, 2),
            vec![
                0.1, 1.0, //
                0.2, 1.1, //
                0.3, 1.2, //
            ],
        )
        .unwrap();

        let decoder = CtcGreedyDecoder::new(CtcGreedyDecoderConfig { blank_id: 1 });
        let dictionary = dictionary_from_tokens(&["a"]);
        let sequences = decoder
            .decode(&logits, &[3], &dictionary)
            .expect("decoding should succeed");

        assert_eq!(sequences[0].text, "");
        assert!(sequences[0].token_indices.is_empty());
        assert_eq!(sequences[0].confidence, 0.0);
    }

    #[test]
    fn error_when_blank_id_out_of_range() {
        let logits = Array3::<f32>::zeros((1, 1, 2));
        let decoder = CtcGreedyDecoder::new(CtcGreedyDecoderConfig { blank_id: 3 });
        let dictionary = dictionary_from_tokens(&["a"]);
        let error = decoder
            .decode(&logits, &[1], &dictionary)
            .expect_err("blank id out of range should error");

        matches!(
            error,
            CtcGreedyDecoderError::BlankIdOutOfRange {
                blank_id: 3,
                class_count: 2
            }
        );
    }
}
