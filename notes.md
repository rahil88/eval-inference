# Notes: eval_inference

What I predicted, measured and learned, one section per step. The README (Step 11) gets written from these.

## Setup
- Machine: Apple M2, 16 GB unified memory (about 12 GB usable by the GPU), about 100 GB/s memory bandwidth
- llama.cpp 1.0.0 (build 12359, commit b1b770d90), installed with Homebrew
- Python 3.12.13 via uv; httpx 0.28.1, pytest 9.2.0
- Model: Qwen2.5-1.5B-Instruct. FP16 is the official GGUF; Q4_K_M I quantized myself with `llama-quantize`

## Step 1: what is GGUF? (in my own words)
-

## Step 2: model file sizes

### Prediction (made before measuring)
| File | Calculation | Predicted |
|---|---|---|
| FP16 | 1.5B × 2 bytes | 3.0 GB |
| Q4_K_M | 1.5B × 4.5 bits ÷ 8 | 0.84 GB |

### Measured
<!-- exact bytes: stat -f "%z %N" models/*.gguf -->
| File | Bytes | GB | Error vs prediction |
|---|---|---|---|
| FP16 | | | |
| Q4_K_M | | | |

### Why the measurements differ from the predictions
- Cause that affects both files (hint: `tie_word_embeddings`):
- Cause that affects only Q4_K_M (hint: the "M"):
