# `ROADMAP.md`

## 🎯 プロジェクト目標 (Overall Goal)

`jingsongliujing/OnnxOCR` [1] の主要機能（DBNet検出 [2], SVTR認識 [3]）を、C/C++ FFIに依存しない **Pure Rust** [4, 5, 6] 環境で再実装する。

## 📊 開発進捗 (Overall Progress)

*   [x] **M0: 技術検証 (Proof of Concept)**
*   [x] **M1: 検出パイプライン (Detection Pipeline)**
*   [ ] **M2: 認識パイプライン (Recognition Pipeline)**
*   [ ] **M3: 統合とAPI (Engine & API Layer)**
*   [ ] **M4: ドキュメントとリリース (Docs & Release)**

---

## 📌 マイルストーン詳細 (Milestones)

### M0: 技術検証 (Proof of Concept)

**目標:** プロジェクトの最大の技術的リスクである「Pure Rust推論エンジン `tract` [4, 7] で `rec.onnx` (SVTR) [3, 8] が実行可能か」を0か1かで判定する。

| ステータス | タスクID       | タスク概要                                                              | 関連ブランチ / イシュー |
| :--------- | :------------- | :---------------------------------------------------------------------- | :---------------------- |
| `[x]`      | `task-poc-001` | `tract-onnx` で `det.onnx` (DBNet [2]) のロードとダミー実行             | `feature/001-poc-det`   |
| `[x]`      | `task-poc-002` | **[最重要]** `tract-onnx` で `rec.onnx` (SVTR [3]) のロードとダミー実行 | `feature/002-poc-rec`   |

### M1: 検出パイプライン (Detection Pipeline)

**目標:** 画像を入力とし、`geo-types::Polygon` [9] のリスト（テキスト領域の座標）を返すモジュールを完成させる。

| ステータス | タスクID       | タスク概要                                                                        | 関連ブランチ / イシュー         |
| :--------- | :------------- | :-------------------------------------------------------------------------------- | :------------------------------ |
| `[x]`      | `task-det-001` | 検出前処理 (`DetPreProcessor`) の実装 (Resize [10], Normalize, NCHW変換 [11, 12]) | `feature/003-det-preproc`       |
| `[x]`      | `task-det-002` | `tract` を使った検出推論の実行                                                    | `feature/004-det-infer`         |
| `[x]`      | `task-det-003` | 検出後処理: `imageproc::find_contours` [13] による輪郭抽出                        | `feature/005-det-post-contours` |
| `[x]`      | `task-det-004` | 検出後処理: `i_overlay::buffering` [14, 9] によるポリゴンオフセット (Unclip) [15] | `feature/006-det-post-offset`   |
| `[x]`      | `task-det-005` | 検出後処理: 座標のスケール復元と `Polygon` への変換                               | `feature/007-det-post-scaling`  |

### M2: 認識パイプライン (Recognition Pipeline)

**目標:** クロップされた画像バッチを入力とし、テキスト文字列のリストを返すモジュールを完成させる。

| ステータス | タスクID       | タスク概要                                                                         | 関連ブランチ / イシュー       |
| :--------- | :------------- | :--------------------------------------------------------------------------------- | :---------------------------- |
| `[x]`      | `task-rec-001` | 認識前処理 (`RecPreProcessor`) の実装 (Crop, Force Resize [3], NCHW変換, Batching) | `feature/008-rec-preproc`     |
| `[x]`      | `task-rec-002` | `tract` を使った認識推論の実行 (バッチ対応)                                        | `feature/009-rec-infer`       |
| `[x]`      | `task-rec-003` | 辞書ファイル (`ppocrv5_dict.txt` [3, 16]) のロード機能実装                         | `feature/010-rec-dict`        |
| `[x]`      | `task-rec-004` | 認識後処理: Pure Rust CTC Greedyデコード [17, 18] のアルゴリズム実装               | `feature/011-rec-post-ctc`    |
| `[ ]`      | `task-rec-005` | 認識後処理: `ndarray::argmax` [19] とCTCデコードの結合                             | `feature/012-rec-post-decode` |

### M3: 統合とAPI (Engine & API Layer)

**目標:** M1とM2を統合し、API設計書で定義した `OcrEngine` と `OcrEngineBuilder` を完成させる。

| ステータス | タスクID       | タスク概要                                                              | 関連ブランチ / イシュー     |
| :--------- | :------------- | :---------------------------------------------------------------------- | :-------------------------- |
| `[ ]`      | `task-api-001` | `OcrEngineBuilder` の実装 (モデルロード [4], 辞書ロード [16]、設定保持) | `feature/013-api-builder`   |
| `[ ]`      | `task-api-002` | `OcrEngine` の実装 (Facade パターン)                                    | `feature/014-api-engine`    |
| `[ ]`      | `task-api-003` | `OcrEngine::run_from_path` の実装 (E2Eパイプライン統合)                 | `feature/015-api-run-path`  |
| `[ ]`      | `task-api-004` | `OcrEngine::run_from_image` の実装 (メモリバッファ対応)                 | `feature/016-api-run-image` |
| `[ ]`      | `task-api-005` | 公開エラー型 `OcrError` の実装と伝達                                    | `feature/017-api-error`     |

### M4: ドキュメントとリリース (Docs & Release)

**目標:** 利用者が迷わず使えるドキュメントを整備し、`crates.io` への公開準備を整える。

| ステータス | タスクID       | タスク概要                                                         | 関連ブランチ / イシュー         |
| :--------- | :------------- | :----------------------------------------------------------------- | :------------------------------ |
| `[ ]`      | `task-doc-001` | `README.md` の作成 (インストール、クイックスタート、API概要)       | `feature/018-doc-readme`        |
| `[ ]`      | `task-doc-002` | `lib.rs` のドキュメントコメント (`cargo doc`) 整備                 | `feature/019-doc-comments`      |
| `[ ]`      | `task-doc-003` | `Cargo.toml` のメタデータ整備 (ライセンス、リポジトリ、キーワード) | `feature/020-cargo-metadata`    |
| `[ ]`      | `task-doc-004` | 結合テスト (`tests/integration_test.rs`) の拡充                    | `feature/021-integration-tests` |