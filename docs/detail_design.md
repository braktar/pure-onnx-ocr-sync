# 詳細設計書 (モジュール設計書): Pure Rust OnnxOCR

作成者: Shion Watanabe  
日付: 2025-11-09  
リポジトリ: http://github.com/siska-tech/pure-onnx-ocr

## 🎯 目的

  * API設計書で定義されたインターフェース（`OcrEngineBuilder`, `OcrEngine`）の内部ロジックを具体化する。
  * 開発者が迷わず実装できるよう、`tract` [1, 2, 3]、`image` [4, 5]、`imageproc` [6, 7, 8, 9]、`i_overlay` [10, 11]、`ndarray` [12, 13, 14, 15, 16, 17] を使用した処理の手順とデータ構造を明確にする。

-----

## 1\. 内部クラス・関数設計

API（`OcrEngineBuilder`, `OcrEngine`）を実現するため、以下のプライベートヘルパー関数と内部データ構造を設計する。

```rust
// --- 内部ヘルパー構造体 (プライベート) ---

/// 検出前処理の結果を保持する内部構造体
struct PreprocessedDetInput {
    /// tract に入力する 形状のテンソル
    tensor: tract_onnx::prelude::Tensor,
    /// リサイズ後の検出モデル入力サイズ (width, height)
    resized_dims: (u32, u32),
    /// 元画像からリサイズ画像へのスケール比
    scale_ratio: f64,
}

/// 認識前処理の結果をバッチで保持する内部構造体
struct PreprocessedRecInput {
    /// tract に入力する 形状のバッチテンソル
    batch_tensor: tract_onnx::prelude::Tensor,
    /// このバッチに対応する元のポリゴン (座標は元画像スケール)
    original_polygons: Vec<geo_types::Polygon>,
}

/// OcrEngine の設定パラメータ (API の OcrConfig と同一)
struct OcrConfig {
    det_limit_side_len: u32,
    det_unclip_ratio: f64,
    det_thresh: f32, // DBNetの二値化閾値 (例: 0.3) [18]
    rec_image_shape: (u32, u32), // (width, height) (例: 320, 48) [19, 20]
    rec_batch_size: usize,
}

// --- OcrEngine 内部のプライベートメソッド ---
impl OcrEngine {
    /// (API `build` から呼ばれる) tract モデルをロードする
    fn load_model(path: &Path) -> Result<RunnableModel, OcrError> {... }

    /// (API `build` から呼ばれる) 辞書ファイルをロードする
    fn load_dictionary(path: &Path) -> Result<(Vec<String>, usize), OcrError> {... }

    /// 検出パイプライン (ステップ 1): 前処理
    fn preprocess_detection(&self, image: &image::DynamicImage) -> Result<PreprocessedDetInput, OcrError> {... }

    /// 検出パイプライン (ステップ 2): 後処理
    fn postprocess_detection(
        &self,
        det_output: &tract_onnx::prelude::Tensor, // 形状
        resized_dims: (u32, u32),
        scale_ratio: f64,
    ) -> Result<Vec<geo_types::Polygon>, OcrError> {... }

    /// 認識パイプライン (ステップ 1): 前処理 (バッチ化を含む)
    fn preprocess_recognition(
        &self,
        image: &image::DynamicImage,
        polygons: &[geo_types::Polygon],
    ) -> Result<Vec<PreprocessedRecInput>, OcrError> {... }

    /// 認識パイプライン (ステップ 2): 後処理 (CTCデコード)
    fn postprocess_recognition(
        &self,
        rec_output: &tract_onnx::prelude::Tensor, // 形状
    ) -> Result<Vec<(String, f32)>, OcrError> {... }
}

// --- 内部ヘルパー関数 (プライベート) ---

/// imageproc の Contour を geo_types の Polygon に変換する
fn contour_to_geo_polygon(contour: &imageproc::contours::Contour<i32>) -> geo_types::Polygon {... }

/// CTC Greedy デコードアルゴリズム
fn ctc_greedy_decode(indices: &[usize], blank_id: usize) -> Vec<usize> {... }
```

## 2\. データ構造

  * **`OcrEngine`**
      * `det_model: RunnableModel` (プライベート): ロード済みの `tract` 検出モデル。
      * `rec_model: RunnableModel` (プライベート): ロード済みの `tract` 認識モデル。
      * `dictionary: Vec<String>` (プライベート): 認識用の文字辞書 [19, 20, 21]。
      * `blank_id: usize` (プライベート): CTCデコード用のブランクID [22, 23] (通常は `dictionary.len()` [24])。
      * `config: OcrConfig` (プライベート): `det_limit_side_len` などの設定値。
  * **`OcrEngineBuilder`**
      * `det_path: Option<PathBuf>`
      * `rec_path: Option<PathBuf>`
      * `dict_path: Option<PathBuf>`
      * `config: OcrConfig` (デフォルト値で初期化)
  * **`OcrResult`**
      * (API設計書で定義済み: `text: String`, `confidence: f32`, `bounding_box: Polygon`) [25]
  * **`OcrError`**
      * (API設計書で定義済み)

## 3\. アルゴリズム・ロジック

### 3.1. `OcrEngineBuilder::build`

1.  `det_path`, `rec_path`, `dict_path` がすべて `Some` であることを検証する。`None` があれば `OcrError::ProcessingError` を返す。
2.  `det_model = OcrEngine::load_model(det_path)` を呼び出す。
3.  `rec_model = OcrEngine::load_model(rec_path)` を呼び出す。
      * **重要 (PoCリスク)**: この `load_model` (内部で `tract_onnx::onnx().model_for_path()` [26, 3] を使用) が、`rec.onnx` (SVTR\_HGNet) [19, 20, 27] のロードに失敗する**致命的リスク**が存在する。`tract` は `LayerNormalization` [28, 29, 30, 31, 32] や `Scan` [33, 34] オペレータのサポートが限定的であるため [1, 35, 33, 36, 37]。
4.  `(dictionary, blank_id) = OcrEngine::load_dictionary(dict_path)` を呼び出す。
5.  `OcrEngine` インスタンスを生成し、`Ok(engine)` を返す。

#### `OcrEngine::load_model` (プライベート)

```rust
fn load_model(path: &Path) -> Result<RunnableModel, OcrError> {
    tract_onnx::onnx()
       .model_for_path(path) // [26, 3]
       .map_err(|e| OcrError::ModelLoadError(format!("Failed to load model {}: {}", path.display(), e)))?
       .into_optimized()
       .map_err(|e| OcrError::ModelLoadError(format!("Failed to optimize model {}: {}", path.display(), e)))?
       .into_runnable()
       .map_err(|e| OcrError::ModelLoadError(format!("Failed to make model runnable {}: {}", path.display(), e)))
}
```

#### `OcrEngine::load_dictionary` (プライベート)

```rust
fn load_dictionary(path: &Path) -> Result<(Vec<String>, usize), OcrError> {
    let content = std::fs::read_to_string(path)
       .map_err(|e| OcrError::IoError(e))?;
    
    // 辞書を行で分割し、"blank" トークン (通常は最初の行だが、PaddleOCRでは最後に追加されることが多い [24]) を扱う
    // PP-OCRv5 の辞書 [19, 20, 21] は通常、文字のみを含み、ブランクは暗黙的に最後のインデックス [22, 23]
    let mut dictionary: Vec<String> = content.lines().map(String::from).collect();
    
    // "blank" トークンを明示的に追加 (リファレンス実装に依存)
    // ここでは、辞書ファイルにはブランクが含まれず、CTCロジックで最後に追加されると仮定する
    // [24] の "CTC-blank is the last element" に基づく
    let blank_id = dictionary.len(); 
    dictionary.push("<blank>".to_string()); // デバッグ用。デコーダロジックは `blank_id` のインデックスのみを使用
    
    Ok((dictionary, blank_id))
}
```

### 3.2. `OcrEngine::run_from_image` (メインロジック)

1.  **検出前処理**: `let prep_det = self.preprocess_detection(image)?` [38, 39, 40, 41, 42] を実行。
2.  **検出推論**: `let det_output = self.det_model.run(tvec!(prep_det.tensor.into()))?` を実行 [26]。`det_output` は `TVec<Arc<Tensor>>`。
3.  **検出後処理**: `let polygons = self.postprocess_detection(det_output.as_ref(), prep_det.resized_dims, prep_det.scale_ratio)?` [43, 18, 44, 45, 46, 47] を実行。
4.  `if polygons.is_empty() { return Ok(Vec::new()); }`
5.  **認識前処理**: `let rec_inputs: Vec<PreprocessedRecInput> = self.preprocess_recognition(image, &polygons)?` [39, 40, 41, 42] を実行。
6.  `all_results: Vec<OcrResult> = Vec::new();`
7.  **認識推論 (バッチ処理)**: `rec_inputs` をループ (または `rayon::par_iter` [12, 14, 16])。
    a.  `let rec_output = self.rec_model.run(tvec!(batch.batch_tensor.into()))?`
    b.  `let decoded_outputs = self.postprocess_recognition(rec_output.as_ref())?` [19, 24]
    c.  `batch.original_polygons`, `decoded_outputs` を `zip` して `OcrResult` [25] を生成し、`all_results` に追加。
8.  `Ok(all_results)` を返す。

### 3.3. `preprocess_detection` (アルゴリズム)

```rust
// 疑似コード
fn preprocess_detection(&self, image: &DynamicImage) -> Result<PreprocessedDetInput, OcrError> {
    let (orig_w, orig_h) = image.dimensions();
    let limit_side_len = self.config.det_limit_side_len as f64; [48, 49, 50]
    
    // アスペクト比を維持してリサイズ [48]
    let (resized_w, resized_h, scale_ratio) = if max(orig_w, orig_h) > limit_side_len {
        if orig_h > orig_w {
            let ratio = limit_side_len / orig_h as f64;
            ( (orig_w as f64 * ratio) as u32, limit_side_len as u32, ratio )
        } else {
            let ratio = limit_side_len / orig_w as f64;
            ( limit_side_len as u32, (orig_h as f64 * ratio) as u32, ratio )
        }
    } else {
        (orig_w, orig_h, 1.0)
    };
    
    // image::resize を使用 [4, 51]
    let resized_image = image.resize(resized_w, resized_h, image::imageops::FilterType::Lanczos3);
    
    // ndarray (HWC) に変換
    let mut array_hwc: Array3<f32> = Array3::from_shape_fn((resized_h as usize, resized_w as usize, 3), |(y, x, c)| {
        resized_image.get_pixel(x as u32, y as u32)[c] as f32
    });
    
    // 正規化 [38, 52]
    array_hwc.par_mapv_inplace(|x| x / 255.0);
    
    // HWC -> NCHW [38, 53, 54, 55, 56, 57]
    let array_nchw = array_hwc.permuted_axes() [55]
                          .insert_axis(Axis(0)) [38, 14]
                          .map_err(|e| OcrError::ProcessingError(e.to_string()))?;
                           
    let tensor = tract_onnx::prelude::Tensor::from_array(array_nchw.as_dyn())?;
    
    Ok(PreprocessedDetInput {
        tensor,
        resized_dims: (resized_w, resized_h),
        scale_ratio,
    })
}
```

### 3.4. `postprocess_detection` (アルゴリズム)

```rust
// 疑似コード
fn postprocess_detection(
    &self,
    det_output: &Tensor, // 形状
    resized_dims: (u32, u32),
    scale_ratio: f64,
) -> Result<Vec<Polygon>, OcrError> {
    
    // 1. テンソルを ndarray (二値化マップ) に変換
    let prob_map_view = det_output.to_array_view::<f32>()?
                                .index_axis(Axis(0), 0)
                                .index_axis(Axis(0), 0); //

    let binary_map: Array2<u8> = prob_map_view.mapv(|p| if p > self.config.det_thresh { 255 } else { 0 }); [18]
    
    // 2. ndarray から image::GrayImage に変換 [58]
    let gray_image = image::GrayImage::from_raw(resized_dims.0, resized_dims.1, binary_map.into_raw_vec())
       .ok_or_else(|| OcrError::ProcessingError("Failed to create GrayImage from binary map".to_string()))?;
        
    // 3. 輪郭抽出 (cv2.findContours の代替) [59, 60]
    let contours: Vec<imageproc::contours::Contour<i32>> = 
        imageproc::contours::find_contours::<i32>(&gray_image); [6, 7, 61, 8, 9, 2, 62]

    let mut polygons: Vec<Polygon> = Vec::new();
    
    // 4. ポリゴン変換とオフセット (Unclip)
    for contour in contours {
        if contour.points.len() < 3 { continue; }
        
        // 4a. imageproc::Contour -> geo_types::Polygon [7]
        let geo_polygon = contour_to_geo_polygon(&contour);
        
        // 4b. オフセット (pyclipper.PyclipperOffset の代替) [63, 18, 64, 45]
        // i_overlay は Pure Rust のポリゴン演算ライブラリ [10, 65, 11, 66, 67]
        let offset_polygons = i_overlay::buffering::buffer_polygon(
            &geo_polygon, 
            self.config.det_unclip_ratio, // [18]
            i_overlay::JoinType::Miter(2.0), // Miter join
            i_overlay::EndType::ClosedPolygon
        );
        
        // 4c. 座標を元画像スケールに戻す
        for poly in offset_polygons {
            let scaled_poly = poly.map_coords(|&(x, y)| (x / scale_ratio, y / scale_ratio));
            // TODO: 面積が小さすぎるポリゴンを除外 [46]
            polygons.push(scaled_poly);
        }
    }
    
    Ok(polygons)
}
```

### 3.5. `preprocess_recognition` (アルゴリズム)

```rust
// 疑似コード
fn preprocess_recognition(
    &self,
    image: &DynamicImage,
    polygons: &[Polygon],
) -> Result<Vec<PreprocessedRecInput>, OcrError> {

    let (rec_w, rec_h) = self.config.rec_image_shape; // (例: 320, 48) [19, 20]

    // rayon を使ってバッチごとに並列処理 [12, 14, 16]
    polygons.par_chunks(self.config.rec_batch_size)
       .map(|poly_chunk| {
            let mut batch_tensors: Vec<Array3<f32>> = Vec::with_capacity(poly_chunk.len());
            
            for polygon in poly_chunk {
                // 簡易的な矩形クロップ (厳密にはアフィン変換が必要)
                let bounding_rect = polygon.bounding_rect();
                let cropped_image = image.crop_imm(
                    bounding_rect.min().x as u32,
                    bounding_rect.min().y as u32,
                    bounding_rect.width() as u32,
                    bounding_rect.height() as u32,
                );
                
                // アスペクト比を無視して強制リサイズ [51, 5]
                let resized_image = cropped_image.resize_exact(rec_w, rec_h, image::imageops::FilterType::Lanczos3); [19, 20, 51]
                
                // HWC -> NCHW 変換と正規化 (preprocess_detection と同様)
                let mut array_hwc: Array3<f32> =... ; // [38, 57]
                array_hwc.par_mapv_inplace(|x| x / 255.0);
                let array_chw = array_hwc.permuted_axes(); [54, 55]
                
                batch_tensors.push(array_chw);
            }
            
            // バッチテンソルを作成
            let batch_array = ndarray::stack(
                Axis(0), 
                &batch_tensors.iter().map(|a| a.view()).collect::<Vec<_>>()
            ).map_err(|e| OcrError::ProcessingError(e.to_string()))?;
            
            Ok(PreprocessedRecInput {
                batch_tensor: tract_onnx::prelude::Tensor::from_array(batch_array.as_dyn())?,
                original_polygons: poly_chunk.to_vec(),
            })
            
        })
       .collect::<Result<Vec<PreprocessedRecInput>, OcrError>>()
}
```

### 3.6. `postprocess_recognition` (アルゴリズム)

```rust
// 疑似コード
fn postprocess_recognition(
    &self,
    rec_output: &Tensor, // 形状
) -> Result<Vec<(String, f32)>, OcrError> {

    let logits_view = rec_output.to_array_view::<f32>()?; //[68, 24, 69]
    let mut results = Vec::with_capacity(logits_view.shape());

    for batch_item_logits in logits_view.outer_iter() { //[14]
        // 1. ArgMax: 各タイムステップで最も確率の高いインデックスを取得
        let indices: Vec<usize> = batch_item_logits
           .argmax_axis(Axis(1)) [14, 16] //
           .map_err(|e| OcrError::ProcessingError(e.to_string()))?
           .to_vec();
            
        // 2. CTC Greedy Decode [24, 70, 22, 71]
        let decoded_indices = ctc_greedy_decode(&indices, self.blank_id);
        
        // 3. 辞書マッピング
        let text: String = decoded_indices.iter()
           .map(|&idx| self.dictionary.get(idx).map_or("", |s| s.as_str()))
           .collect();
            
        // 4. 信頼度計算 (簡易版: ここではダミー値)
        // (ロジックの複雑化を避けるため、API設計書の定義に従い 0.0-1.0 の値を設定する)
        let confidence = 1.0; // TODO: 本来は softmax 確率の平均を計算
        
        results.push((text, confidence));
    }
    
    Ok(results)
}

/// CTC Greedy Decode (プライベートヘルパー)
fn ctc_greedy_decode(indices: &[usize], blank_id: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut prev_index = usize::MAX;
    
    for &index in indices {
        // 1. blank でない [22, 23]
        // 2. 前の文字と重複していない [70]
        if index!= blank_id && index!= prev_index {
            result.push(index);
        }
        prev_index = index;
    }
    result
}
```

## 4\. エラー処理の詳細

  * **`OcrEngineBuilder::build`**:
      * **検知**: `det_path`, `rec_path`, `dict_path` が `None`。
      * **伝達**: `Err(OcrError::ProcessingError)` を返す。
      * **検知**: `std::fs::read_to_string` が失敗。
      * **伝達**: `Err(OcrError::IoError)` を返す。
      * **検知**: `tract_onnx::onnx().model_for_path()` または `.into_runnable()` が失敗。
      * **伝達**: `tract::Error` を `Err(OcrError::ModelLoadError)` に変換して返す。
  * **`OcrEngine::run_from_path`**:
      * **検知**: `image::open` が失敗。
      * **伝達**: `image::ImageError` を `Err(OcrError::ImageError)` に変換して返す。
  * **`OcrEngine::run_from_image`**:
      * **検知**: `self.det_model.run()` または `self.rec_model.run()` が失敗。
      * **伝達**: `tract::Error` を `Err(OcrError::InferenceError)` に変換して返す。
      * **検知**: `preprocess_*` / `postprocess_*` 内の `ndarray` 操作 (`permuted_axes`, `stack`, `argmax_axis`) が失敗。
      * **伝達**: `ndarray::ShapeError` を `Err(OcrError::ProcessingError)` に変換して返す。
      * **検知**: `imageproc::contours` や `i_overlay::buffering` が論理的に失敗 (例: `GrayImage::from_raw` が `None` を返す)。
      * **伝達**: `Err(OcrError::ProcessingError)` を返す。
  * **`OcrError`** (API設計書より):
      * `IoError(std::io::Error)`
      * `ImageError(image::ImageError)`
      * `ModelLoadError(String)` (Tractのエラーメッセージを含む)
      * `InferenceError(String)` (Tractのエラーメッセージを含む)
      * `ProcessingError(String)` (Ndarray, Imageproc, iOverlay関連のエラー)