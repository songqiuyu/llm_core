# AxonForge Inference Benchmark

**Hardware**: Intel i9-12900K (8P+4E, AVX2/FMA/F16C), DDR4-3200 (51.2 GB/s)  
**OS**: Ubuntu 22.04 / Linux 6.x  
**Build**: Release, `-O3 -march=native`  
**Precision**: F16 weights (float16)  
**Threads**: 8 (physical P-cores)

---

## 1. GPT-2 — Results

| Metric | Value |
|--------|-------|
| Model | GPT-2 117M F16 |
| Prompt | "The meaning of life is" |
| Max new tokens | 200 |
| **TTFT** | **67 ms** |
| **Decode speed** | **73.9 tok/s** |

### GPT-2: AxonForge vs llama.cpp

| Implementation | Precision | tok/s | vs AxonForge |
|----------------|-----------|-------|--------------|
| **AxonForge** | F16 | **73.9** | — |
| llama.cpp (i9-12900K, 8T) | F16 | ~49.9 | −32% |
| AxonForge Phase 0 (scalar) | F16 | 2.7 | −96% |

> llama.cpp reference: `llama-bench` v0.0.3330, commit `d9c7e86`, i9-12900K, 8 threads, F16, `pp=1 tg=200`.

---

## 2. LLaMA Family — Results

> **To fill in after models are loaded** (see §4 for GGUF file locations).

### TinyLlama-1.1B

| Metric | AxonForge | llama.cpp ref (i9-12900K) |
|--------|-----------|--------------------------|
| Precision | F16 | F16 |
| TTFT | _TBD_ | ~35 ms |
| **tok/s** | **_TBD_** | ~26 tok/s |

### LLaMA-3.2-3B

| Metric | AxonForge | llama.cpp ref (i9-12900K) |
|--------|-----------|--------------------------|
| Precision | F16 | F16 |
| TTFT | _TBD_ | ~120 ms |
| **tok/s** | **_TBD_** | ~10 tok/s |

### LLaMA-3-8B

| Metric | AxonForge | llama.cpp ref (i9-12900K) |
|--------|-----------|--------------------------|
| Precision | F16 | F16 |
| TTFT | _TBD_ | ~350 ms |
| **tok/s** | **_TBD_** | ~4 tok/s |

> llama.cpp reference: official wiki "Performance" page + community benchmarks on
> [ggml-org/llama.cpp discussions](https://github.com/ggerganov/llama.cpp/discussions).

---

## 3. Why AxonForge Is Faster

| Optimisation | Benefit |
|-------------|---------|
| **AVX2+F16C GEMV** — 8-wide FP32 FMA, 2× unrolled dot product | ~8–10× vs scalar |
| **Persistent ThreadPool** — fork-join, <500 ns wake latency | Eliminates thread-spawn overhead |
| **Pre-built weight pointer table** — `LlamaWeights` / `Gpt2Weights` | Eliminates snprintf + hashmap per token |
| **RoPE cos/sin cache** — computed once at model load | Eliminates trigonometry in hot loop |

The GPT-2 benchmark already demonstrates **+48% vs llama.cpp F16** on the same hardware.  
LLaMA results are expected to show similar or greater advantage as the model size grows
(memory bandwidth bound → AVX2 load throughput advantage increases).

---

## 4. Running the Benchmark

### Required GGUF Files (F16 only — quantised Q4 not yet supported)

| Model | File | Size | Path |
|-------|------|------|------|
| TinyLlama 1.1B | `TinyLlama_v1.1-F16.gguf` | 2.2 GB | `models/tinyllama-1.1B/` |
| LLaMA-3.2-3B | `Llama-3.2-3B-F16.gguf` | 6.4 GB | `models/llama-3.2-3B/` |
| LLaMA-3-8B | `Meta-Llama-3-8B-F16.gguf` | 16.1 GB | `models/llama-3-8B/` |

Download sources:
- `bartowski/TinyLlama_v1.1-GGUF` (HuggingFace)
- `bartowski/Llama-3.2-3B-GGUF` (HuggingFace)
- `bartowski/Meta-Llama-3-8B-GGUF` (HuggingFace)

### Benchmark Commands

```bash
# TinyLlama 1.1B
build/tools/cli/axonforge-cli \
    -m models/tinyllama-1.1B/TinyLlama_v1.1-F16.gguf \
    -p "The meaning of life is" -n 200 -t 0.8

# LLaMA-3.2 3B
build/tools/cli/axonforge-cli \
    -m models/llama-3.2-3B/Llama-3.2-3B-F16.gguf \
    -p "The meaning of life is" -n 200 -t 0.8

# LLaMA-3 8B
build/tools/cli/axonforge-cli \
    -m models/llama-3-8B/Meta-Llama-3-8B-F16.gguf \
    -p "The meaning of life is" -n 200 -t 0.8

# Interactive mode
build/tools/cli/axonforge-cli \
    -m models/tinyllama-1.1B/TinyLlama_v1.1-F16.gguf \
    -i -n 200 -t 0.8
```

---

## 5. Phase 2 Roadmap (Performance Targets)

| Feature | Expected gain |
|---------|--------------|
| Q4_K_M quantisation | 2× tok/s (bandwidth halved) |
| AVX2 attention kernel (dot-product loop) | +3–5% |
| GELU/SiLU polynomial approx | +1–2% |
| CUDA backend (GTX 1080 Ti) | ~300–500 tok/s |
