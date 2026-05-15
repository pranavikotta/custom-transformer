# Custom GPT-Style Transformer from Scratch

An end-to-end generative language model architected and trained from scratch. This project implements a custom Byte-Pair Encoding (BPE) tokenizer and a GPT-style autoregressive Transformer, optimized for embedded deployment and edge computing.

## Key Features
* **Custom BPE Tokenizer:** Developed from scratch to compress text data, allowing a wider context window without increasing compute footprint.
* **Transformer Architecture:** Multi-layer decoder-only Transformer incorporating multi-head causal self-attention and layer normalization.
* **Production Deployment:** Exported to an optimized, platform-agnostic **ONNX** graph layout.

## Performance Metrics & Benchmark Results

The model was trained on the **TinyStories** dataset and evaluated using a standalone lightweight `onnxruntime` engine:

| Metric | Evaluation Value | Engineering Interpretation |
| :--- | :--- | :--- |
| **Final Val Loss** | 2.22 | Successful convergence without overfitting. |
| **Perplexity (PPX)** | **10.16** | Sharp distribution; limits token predictions to ~10 logical choices. |
| **Compression Ratio** | **3.05x** | BPE tokenization packs ~780 raw characters into a 256 context window. |
| **Top-1 Accuracy** | 45.80% | High fidelity to standard grammar structures. |
| **Top-5 Accuracy** | **71.62%** | High semantic coherence; correct word options consistently sit in top choices. |
| **Inference Speed** | **86.25 tokens/sec** | High-speed, low-latency execution optimized for real-time edge hardware. |
| **Diversity Ratio** | 0.58 | Healthy vocabulary variance preventing permanent generation looping. |

---

## Architectural Insights & Footprint Optimization

### Model Size
* **FP32 ONNX Model Weight:** **0.31 MB** (310 KB)

### Why Quantization Was Omitted
During optimization steps, dynamic `INT8` quantization was applied to evaluate edge compatibility. The resulting file grew to **1.90 MB** (a ~6x size expansion). 

**Engineering Analysis:**
1.  **Metadata Overhead:** Because the baseline network is incredibly compact (**0.31 MB**), the structural scaling constants and zero-point tensors introduced by quantization far outweighed the memory footprint of the actual layer weights.
2.  **Deployment Verdict:** At 310 KB, the FP32 high-precision model is already optimized enough to fit comfortably directly onto the flash memory of advanced microcontrollers (e.g., STM32, ESP32) or a Raspberry Pi. Therefore, the native **FP32 configuration** was preserved to ensure maximum structural precision and performance efficiency.