# Reference Library (English Edition)

Author: Shion Watanabe  
Date: 2025-11-09  
Repository: http://github.com/siska-tech/pure-onnx-ocr

This bibliography mirrors the Japanese reference list and is grouped for easier navigation by international contributors.

## Core Projects & Papers

1. PaddleOCR project home – <https://github.com/PaddlePaddle/PaddleOCR>
2. OnnxOCR reference implementation – <https://github.com/jingsongliujing/OnnxOCR>
3. DBNet (Differentiable Binarization) paper – <https://arxiv.org/abs/2003.1035>
4. SVTR / PP-OCRv5 recognition overview – <http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html>
5. PaddleOCR pipeline usage guide – <http://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html>

## ONNX Models & Discussions

1. PP-OCRv5 detection model – <https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det>
2. PP-OCRv5 recognition model – <https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec>
3. PaddleOCR ONNX export discussions – <https://github.com/PaddlePaddle/PaddleOCR/discussions/14572>
4. PaddleOCR ONNX issues tracker – <https://github.com/PaddlePaddle/PaddleOCR/issues/16476>
5. ONNX operator specifications – <https://onnx.ai/onnx/operators/>

## Rust Ecosystem

| Area                | Resource                                                               |
| ------------------- | ---------------------------------------------------------------------- |
| ONNX inference      | `tract-onnx` crate – <https://crates.io/crates/tract-onnx>             |
| Tensor utilities    | `ndarray` crate – <https://crates.io/crates/ndarray>                   |
| Image processing    | `image` crate – <https://crates.io/crates/image>                       |
| Contour detection   | `imageproc` crate – <https://github.com/image-rs/imageproc>            |
| Polygon buffering   | `i_overlay` crate – <https://docs.rs/i_overlay>                        |
| Geometry primitives | `geo-types` crate – <https://docs.rs/geo>                              |
| Polygon clipping    | `polygon_clipping` crate – <https://crates.io/crates/polygon_clipping> |

## Algorithms & Techniques

- CTC greedy decoding background – <https://distill.pub/2017/ctc/>
- Vatti polygon clipping algorithm – <https://en.wikipedia.org/wiki/Vatti_clipping_algorithm>
- Greiner–Hormann polygon clipping – <https://www.inf.usi.ch/hormann/papers/Greiner.1998.ECO.pdf>
- PaddleOCR DB post-processing source – <https://gitee.com/paddlepaddle/PaddleOCR/blob/release/2.6/ppocr/postprocess/db_postprocess.py>
- Image resizing best practices in Rust – <https://rust.code-maven.com/resize-image>

## Tutorials & Community Notes

- PaddleOCR quick introduction – <https://learnopencv.com/optical-character-recognition-using-paddleocr/>
- PaddleOCR with ONNX Runtime tutorial – <https://fxis.ai/edu/how-to-work-with-onnx-models-translated-from-paddleocr/>
- Rust community discussion on pure Rust OCR – <https://www.reddit.com/r/rust/comments/lnuaau/pure_rust_tag_discussion/>
- Rust image processing tutorial – <https://www.freecodecamp.org/news/rust-tutorial-naive-star-detector-for-images/>

## Tooling & Deployment

- crate2nix documentation for Nix builds – <https://nix-community.github.io/crate2nix/00_guides/80_using_a_rust_overlay/>
- OpenVINO notebook for PaddleOCR – <https://docs.openvino.ai/2024/notebooks/paddle-ocr-webcam-with-output.html>

This list is not exhaustive; see the Japanese edition (`docs/references.md`) for the full citation index. Contributions adding clarifying notes or new resources are welcome.

