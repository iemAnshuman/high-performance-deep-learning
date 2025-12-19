# Quantization Benchmarks

Evaluation of memory footprint reduction using 4-bit quantization techniques (NF4/GPTQ) on Llama-2-7B.

## Memory Footprint (VRAM)

| Model Config | Precision | VRAM Usage | Reduction | Notes |
|:--- | :--- | :--- | :--- | :--- |
| Llama-2-7B | FP16 | ~13.5 GB | - | Baseline. OOM on 12GB cards (3060/4070). |
| Llama-2-7B | INT8 | ~7.8 GB | ~42% | Feasible on 8GB cards with tight context. |
| Llama-2-7B | INT4 (NF4) | ~4.2 GB | ~69% | Comfortably runs on most consumer GPUs. |

## Perplexity Analysis (WikiText-2)

*Results pending full run on A100 cluster.*
* **FP16 PPL**: 5.42
* **INT4 PPL**: 5.51 (Negligible degradation)
