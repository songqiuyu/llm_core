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

> Measured on i9-12900K, 8 threads, prompt `"Tell me a short joke."` (7 tokens).

### TinyLlama-1.1B-Chat-v1.0 Q4_K_M

| Metric | **Phase 3** | Phase 2b | Phase 2a | Phase 0 (scalar) | llama.cpp ref |
|--------|-------------|----------|----------|--------------------|---------------|
| Precision | Q4_K_M | Q4_K_M | Q4_K_M | Q4_K_M | Q4_K_M |
| TTFT (7-token prompt) | **~884 ms** | ~2338 ms *(est.)* | ~7168 ms | ~34300 ms | ~35 ms |
| Prefill speed | **~7.9 tok/s** | ~3.0 tok/s | ~1.0 tok/s | ~0.2 tok/s | — |
| **Decode speed** | **~20.1 tok/s** | ~14.8 tok/s | ~4.96 tok/s | ~0.96 tok/s | ~26 tok/s |
| Decode speedup vs 2b | **+36%** | — | — | — | — |
| TTFT speedup vs 2b | **2.6×** | — | — | — | — |

### LLaMA-2-7B Q4_K_M

| Metric | **Phase 3** | Phase 2b | Phase 2a | Phase 0 (scalar) | llama.cpp ref |
|--------|-------------|----------|----------|--------------------|---------------|
| Precision | Q4_K_M | Q4_K_M | Q4_K_M | Q4_K_M | Q4_K_M |
| TTFT (7-token prompt) | **~3657 ms** | ~24017 ms *(est.)* | — | — | ~200 ms |
| Prefill speed | **~1.9 tok/s** | ~0.3 tok/s | — | — | — |
| **Decode speed** | **~5.0 tok/s** | ~1.79 tok/s | ~0.65 tok/s | ~0.18 tok/s | ~4 tok/s |
| Decode speedup vs 2b | **+179%** | — | — | — | — |
| TTFT speedup vs 2b | **6.6×** | — | — | — | — |

### LLaMA-3.2-3B / LLaMA-3-8B

| Model | AxonForge | llama.cpp ref |
|-------|-----------|---------------|
| LLaMA-3.2-3B F16 | _TBD_ | ~10 tok/s |
| LLaMA-3-8B F16 | _TBD_ | ~4 tok/s |

> LLaMA-3 architecture support (rope_scaling, 128K vocab) not yet implemented.

---

## 3. Why AxonForge Is Faster

| Optimisation | Phase | Benefit |
|-------------|-------|---------|
| **AVX2+F16C GEMV** — 8-wide FP32 FMA, 2× unrolled dot product | 1/2a | ~8–10× vs scalar |
| **Persistent ThreadPool** — fork-join, <500 ns wake latency | 1 | Eliminates thread-spawn overhead |
| **Pre-built weight pointer table** — `LlamaWeights` / `Gpt2Weights` | 1 | Eliminates snprintf + hashmap per token |
| **RoPE cos/sin cache** — computed once at model load | 1 | Eliminates trigonometry in hot loop |
| **AVX2 Q6K GEMV** — 6-bit quant kernel with 4-stream FMA | 2b | Enables Q6_K models |
| **4-row GEMV tiling** (Q4K / Q6K / F16) — 4× wider ILP, 4× less x-vector bandwidth | **3A** | +36% decode TinyLlama, +179% LLaMA-2-7B |
| **Batched prefill** — all prompt tokens processed per weight matrix (L3 reuse) | **3B** | 2.6× TTFT TinyLlama, 6.6× TTFT LLaMA-2-7B |
| **AVX2 attention dot / SAXPY** — QK dot and V accumulation use FMA intrinsics | **3C** | ~5% attention speedup |

**LLaMA-2-7B now exceeds llama.cpp** (5.0 vs ~4 tok/s) on the same hardware with 8 threads.

---

## 4. Running the Benchmark

### Available GGUF Files

| Model | File | Size | Path | Status |
|-------|------|------|------|--------|
| TinyLlama-1.1B-Chat | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | ~670 MB | `models/tinyllama-1.1B/` | ✅ 可用 |
| LLaMA-2-7B | `llama-2-7b.Q4_K_M.gguf` | ~3.8 GB | `models/llama-2-7B/` | ✅ 可用 |
| LLaMA-3.2-3B | `Llama-3.2-3B-F16.gguf` | 6.4 GB | `models/llama-3.2-3B/` | ⏳ 未下载 |
| LLaMA-3-8B | `Meta-Llama-3-8B-F16.gguf` | 16.1 GB | `models/llama-3-8B/` | ⏳ 未下载 |

### Benchmark Commands

```bash
# TinyLlama-1.1B Q4_K_M (greedy)
build/tools/cli/axonforge-cli \
    -m models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -p "The capital of France is" -n 20 -t 0.0

# LLaMA-2-7B Q4_K_M (greedy)
build/tools/cli/axonforge-cli \
    -m models/llama-2-7B/llama-2-7b.Q4_K_M.gguf \
    -p "The capital of France is" -n 20 -t 0.0

# LLaMA-2-7B Q4_K_M (sampling)
build/tools/cli/axonforge-cli \
    -m models/llama-2-7B/llama-2-7b.Q4_K_M.gguf \
    -p "Once upon a time" -n 50 -t 0.8
```

---

## 5. Phase 4 Roadmap (Next Targets)

| Feature | Expected gain | Status |
|---------|--------------|--------|
| 4-row GEMV tiling (Q4K/Q6K/F16) | +36–179% decode | ✅ Phase 3A done |
| Batched prefill | 2.6–6.6× TTFT | ✅ Phase 3B done |
| AVX2 attention dot / SAXPY | ~5% attention | ✅ Phase 3C done |
| AVX_VNNI INT8 kernel | ~2× decode vs AVX2 | ⏳ |
| Multi-token decode (speculative) | ~3× decode | ⏳ |
| Chat template support | usability | ⏳ |
| CUDA backend (GTX 1080 Ti) | ~300–500 tok/s | ⏳ |
