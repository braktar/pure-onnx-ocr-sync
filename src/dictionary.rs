use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
};

/// Errors that can occur while loading or using the recognition dictionary.
#[derive(Debug)]
pub enum DictionaryError {
    /// The provided path could not be read.
    Io { source: io::Error, path: PathBuf },
    /// The dictionary file did not contain any entries.
    EmptyDictionary { path: PathBuf },
    /// The dictionary file contained a duplicate token.
    DuplicateEntry {
        path: PathBuf,
        line_number: usize,
        token: String,
    },
}

impl std::fmt::Display for DictionaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DictionaryError::Io { source, path } => {
                write!(f, "failed to read dictionary file {:?}: {}", path, source)
            }
            DictionaryError::EmptyDictionary { path } => {
                write!(
                    f,
                    "dictionary file {:?} does not contain any valid entries",
                    path
                )
            }
            DictionaryError::DuplicateEntry {
                path,
                line_number,
                token,
            } => {
                write!(
                    f,
                    "dictionary file {:?} contains duplicate entry {:?} on line {}",
                    path, token, line_number
                )
            }
        }
    }
}

impl std::error::Error for DictionaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DictionaryError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Recognition dictionary wrapping the PaddleOCR vocabulary.
#[derive(Debug, Clone)]
pub struct RecDictionary {
    tokens: Vec<String>,
    reverse: HashMap<String, usize>,
}

impl RecDictionary {
    /// Loads a dictionary from a UTF-8 encoded text file.
    ///
    /// Each non-empty line is treated as a token. Lines containing only
    /// whitespace are ignored. Duplicate tokens result in an error.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, DictionaryError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|source| DictionaryError::Io {
            source,
            path: path.to_path_buf(),
        })?;

        let mut tokens = Vec::new();
        let mut reverse = HashMap::new();

        for (line_number, raw_line) in contents.lines().enumerate() {
            let token = raw_line.trim();
            if token.is_empty() {
                continue;
            }

            if reverse.contains_key(token) {
                return Err(DictionaryError::DuplicateEntry {
                    path: path.to_path_buf(),
                    line_number: line_number + 1,
                    token: token.to_string(),
                });
            }

            reverse.insert(token.to_string(), tokens.len());
            tokens.push(token.to_string());
        }

        if tokens.is_empty() {
            return Err(DictionaryError::EmptyDictionary {
                path: path.to_path_buf(),
            });
        }

        Ok(Self { tokens, reverse })
    }

    /// Returns the number of entries in the dictionary.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if the dictionary has no entries.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the token for the given index.
    pub fn token(&self, index: usize) -> Option<&str> {
        self.tokens.get(index).map(|value| value.as_str())
    }

    /// Returns the index for a given token, if it exists.
    pub fn index_of(&self, token: &str) -> Option<usize> {
        self.reverse.get(token).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_file(prefix: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}.txt", prefix, timestamp))
    }

    #[test]
    fn load_dictionary_successfully() {
        let path = unique_temp_file("dict_success");
        fs::write(
            &path,
            "a\n\nb\n c \n# comment-like text should still be taken literally\n",
        )
        .unwrap();

        let dictionary = RecDictionary::from_path(&path).unwrap();

        assert_eq!(dictionary.len(), 4);
        assert_eq!(dictionary.token(0), Some("a"));
        assert_eq!(dictionary.token(1), Some("b"));
        assert_eq!(dictionary.token(2), Some("c"));
        assert_eq!(
            dictionary.token(3),
            Some("# comment-like text should still be taken literally")
        );
        assert_eq!(dictionary.index_of("c"), Some(2));
        assert!(dictionary.index_of("missing").is_none());

        fs::remove_file(path).ok();
    }

    #[test]
    fn error_on_empty_dictionary() {
        let path = unique_temp_file("dict_empty");
        fs::write(&path, "   \n\n\t").unwrap();

        let error = RecDictionary::from_path(&path).unwrap_err();
        match error {
            DictionaryError::EmptyDictionary { .. } => {}
            _ => panic!("expected EmptyDictionary error, got {:?}", error),
        }

        fs::remove_file(path).ok();
    }

    #[test]
    fn error_on_duplicate_entry() {
        let path = unique_temp_file("dict_dup");
        fs::write(&path, "foo\nbar\nfoo\n").unwrap();

        let error = RecDictionary::from_path(&path).unwrap_err();
        match error {
            DictionaryError::DuplicateEntry { line_number, .. } => {
                assert_eq!(line_number, 3);
            }
            _ => panic!("expected DuplicateEntry error, got {:?}", error),
        }

        fs::remove_file(path).ok();
    }
}
