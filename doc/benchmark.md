# AxonForge Inference Benchmark

**Hardware**: Intel i9-12900K (12P+4E, AVX2/FMA/F16C/AVX-VNNI), DDR4-3200 (51.2 GB/s)  
**OS**: Ubuntu 22.04 / Linux 6.x  
**Build**: Release (`-DCMAKE_BUILD_TYPE=Release`), `-O3 -mavx2 -mfma`  
**Threads**: 16（默认；15 workers + 1 main，spinwait pool）

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

| Metric | **Phase 10** | Phase 9 | Phase 8 | Phase 7 | Phase 6 | Phase 5 | Phase 0 (scalar) | llama.cpp ref |
|--------|-------------|---------|---------|---------|---------|---------|------------------|---------------|
| Threads | **16** | 14 | 14 | 14 | 14 | 14 | 1 | 8 |
| TTFT (warm, 8-token) | **~285 ms** ¹ | ~300 ms | ~114 ms ² | ~114 ms | ~100 ms | ~190 ms | ~34300 ms | ~35 ms |
| Prefill speed (T=89) | **~102 tok/s** | ~102 tok/s | ~102 tok/s | ~102 tok/s | ~67 tok/s | ~40 tok/s | ~0.2 tok/s | ~400 tok/s |
| **Decode speed** | **~75 tok/s** | ~69–70 tok/s | ~70–71 tok/s | ~68–70 tok/s | ~65.7 tok/s | ~25.5 tok/s | ~0.96 tok/s | ~77 tok/s |
| vs llama.cpp decode | **~97%** | ~91% | ~91% | ~90% | ~85% | ~36% | ~1.4% | — |
| 内存带宽利用率 | **~73%** ³ | ~90% | ~93% | ~91% | ~88% | ~34% | ~1.3% | ~100% |

> ¹ Phase 10 TTFT 包含 Q4K→r8 重排（~130 ms）+ 线程池创建（~20 ms）；同进程第 2 次调用命中缓存时 TTFT 降至 ~150 ms。  
> ² Phase 8 记录没有包含线程池创建开销，与 Phase 10 的重排计数方法不同。  
> ³ Phase 10 带宽利用率下降原因：j=16 超过 DDR4 实际带宽极限，紧带宽场景下线程同步开销相对占比更高。

### LLaMA-2-7B Q4_K_M

| Metric | **Phase 8** | Phase 6 | Phase 5 | Phase 4 | Phase 3 | llama.cpp ref |
|--------|-------------|---------|---------|---------|---------|---------------|
| TTFT (warm, 4-token) | **~1.5 s** ¹ | ~1.5 s | ~2.4 s | ~2.4 s | ~3.7 s | ~200 ms |
| Prefill speed | **~2.5 tok/s** | ~2.5 tok/s | ~2.9 tok/s | ~2.9 tok/s | ~1.9 tok/s | — |
| **Decode speed** | **~11.5 tok/s** | ~10.9 tok/s | ~6.4 tok/s | ~6.1 tok/s | ~5.0 tok/s | ~15 tok/s |
| vs Phase 6 | **+5%** | — | — | — | — | — |

> ¹ Cold load adds ~10 s for 7B Q4K repacking.

### LLaMA-3.2-3B / LLaMA-3-8B

| Model | AxonForge | llama.cpp ref |
|-------|-----------|---------------|
| LLaMA-3.2-3B F16 | _TBD_ | ~10 tok/s |
| LLaMA-3-8B F16 | _TBD_ | ~4 tok/s |

> LLaMA-3 architecture support (rope_scaling, 128K vocab) not yet implemented.

---

## 3. Qwen / DeepSeek / Hunyuan Smoke Results

第一阶段目标是端到端跑通，不做 llama.cpp 全量性能对标。Qwen2.5 和 DeepSeek-R1-Distill-Qwen 复用现有 x86 Q4_K_M / Q6_K / F16 kernel、Q8 activation、Q4K r8 repack 和 spinwait ThreadPool，并补充 GGUF Q5_0 / Q8_0 权重读取。Q8_0 权重在 decode 阶段使用 AVX2 int8 dot，主要加速 Qwen/DeepSeek 的 LM head。

| Model | Precision | Prompt | Result |
|-------|-----------|--------|--------|
| Qwen2.5-0.5B-Instruct | Q4_K_M | `"The capital of France is"` | PASS: 64 tokens, decode ~30.8 tok/s, TTFT ~352 ms, RSS ~450 MB |
| DeepSeek-R1-Distill-Qwen-1.5B | Q4_K_M | `"What is 2+2?"` | PASS: generated “four”, 32 tokens, decode ~50.2 tok/s, TTFT ~552 ms, RSS ~1.57 GB |
| Qwen2.5-3B-Instruct | Q4_K_M | `"The capital of France is"` | TBD: 3B target, expected to reuse `qwen2` path |
| Qwen2.5-Coder-3B-Instruct | Q4_K_M | Python Fibonacci prompt | TBD: 3B code target, expected to reuse `qwen2` path |
| Qwen3-4B | Q4_K_M | `"What is 2+2? /no_think"` | TBD: experimental `qwen3` dense path with Q/K norm |
| Hunyuan-1.8B-Instruct | Q4_K_M | `"/no_think What is the capital of France?"` | TBD: experimental `hunyuan-dense` path with RoPE-after Q/K norm |

> Qwen2 r8 repack is speed-first: DeepSeek gains decode throughput (~31 → ~50 tok/s) but uses extra repack memory. For a strict low-memory mode, disable Qwen2 r8 in `build_weights()` and keep the F32 activation fallback.

Commands:

```bash
build/tools/cli/axonforge-cli \
    -m models/qwen2.5-0.5B/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -p "The capital of France is" -n 16 -t 0.0

build/tools/cli/axonforge-cli \
    -m models/deepseek-r1-distill-qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B.Q4_K_M.gguf \
    -p "What is 2+2?" -n 32 -t 0.0

build/tools/cli/axonforge-cli \
    -m models/qwen2.5-3B/qwen2.5-3b-instruct-q4_k_m.gguf \
    -p "The capital of France is" -n 64 -t 0.0 -V

build/tools/cli/axonforge-cli \
    -m models/qwen2.5-coder-3B/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
    -p "Write a Python function that returns the nth Fibonacci number." -n 96 -t 0.0 -V

build/tools/cli/axonforge-cli \
    -m models/qwen3-4B/Qwen3-4B-Q4_K_M.gguf \
    -p "What is 2+2? /no_think" -n 64 -t 0.7 --top-k 20 --top-p 0.8 --rep-penalty 1.5 -V

build/tools/cli/axonforge-cli \
    -m models/hunyuan-1.8B/hunyuan-1.8b-instruct-q4_k_m.gguf \
    -p "/no_think What is the capital of France?" -n 64 -t 0.0 -V
```

公平 benchmark 规则：AxonForge 与 llama.cpp 必须使用同一 GGUF、同一 prompt、同一线程数、同一 context cap、同一生成 token 数，并分别记录 TTFT、prefill tok/s、decode tok/s、RSS。单次 CLI smoke 不再作为性能对标结论。

llama.cpp baseline command:

```bash
./llama.cpp/build/bin/llama-cli \
    -m models/hunyuan-1.8B/hunyuan-1.8b-instruct-q4_k_m.gguf \
    -p "/no_think What is the capital of France?" -n 64 -t 0.0 \
    -c 4096 -t 16 --no-warmup --perf
```

---

## 4. CUDA / RTX 4080 Benchmark Plan

CUDA 后端使用 `AXONFORGE_ENABLE_CUDA=ON` 构建，RTX 4080 推荐 `AXONFORGE_CUDA_ARCH=89`。v1 已接入 backend lifecycle、device storage、CUDA tensor copy、RMSNorm/SwiGLU/RoPE 以及 Q4_K_M×F32 fused dequant GEMV 原型 kernel；完整 GPU-resident Qwen2 decoder forward 仍处于实验阶段。

| Model | Backend | Prompt | TTFT | Prefill tok/s | Decode tok/s | VRAM | Status |
|-------|---------|--------|------|---------------|--------------|------|--------|
| Qwen2.5-3B-Instruct Q4_K_M | AxonForge CUDA | `"The capital of France is"` | TBD | TBD | TBD | TBD | build-cuda server validation pending |
| Qwen2.5-3B-Instruct Q4_K_M | llama.cpp CUDA | same | TBD | TBD | TBD | TBD | fair baseline pending |

AxonForge CUDA:

```bash
build-cuda/tools/cli/axonforge-cli \
    -b cuda \
    -m models/qwen2.5-3B/qwen2.5-3b-instruct-q4_k_m.gguf \
    -p "The capital of France is" \
    -n 64 -t 0.0 -V
```

llama.cpp CUDA baseline:

```bash
./llama.cpp/build/bin/llama-cli \
    -m models/qwen2.5-3B/qwen2.5-3b-instruct-q4_k_m.gguf \
    -p "The capital of France is" \
    -n 64 -t 0.0 \
    -c 4096 -ngl 999
```

公平性要求：同一 GGUF、同一 prompt、同一 context cap、同一生成 token 数；CPU baseline 记录 RSS，CUDA baseline 记录 peak VRAM。

---

## 5. Why AxonForge Is Faster

| Optimisation | Phase | Benefit |
|-------------|-------|---------|
| **AVX2+F16C GEMV** — 8-wide FP32 FMA, 2× unrolled dot product | 1/2a | ~8–10× vs scalar |
| **Persistent ThreadPool** — fork-join, <500 ns wake latency | 1 | Eliminates thread-spawn overhead |
| **Pre-built weight pointer table** — `LlamaWeights` / `Gpt2Weights` | 1 | Eliminates snprintf + hashmap per token |
| **RoPE cos/sin cache** — computed once at model load | 1 | Eliminates trigonometry in hot loop |
| **AVX2 Q6K GEMV** — 6-bit quant kernel with 4-stream FMA | 2b | Enables Q6_K models |
| **4-row GEMV tiling** (Q4K / Q6K / F16) — 4× wider ILP, 4× less x-vector bandwidth | 3A | +36% decode TinyLlama, +179% LLaMA-2-7B |
| **Batched prefill** — all prompt tokens processed per weight matrix (L3 reuse) | 3B | 2.6× TTFT TinyLlama, 6.6× TTFT LLaMA-2-7B |
| **AVX2 attention dot / SAXPY** — QK dot and V accumulation use FMA intrinsics | 3C | ~5% attention speedup |
| **AVX-VNNI Q4K INT8 path** — `dpbusd` replaces `maddubs+madd` | 5 | +9% decode |
| **Release build** — fix Debug(-O0) regression on forward pass & thread pool lambda | **6** | +47% decode |
| **Spinwait thread pool** — atomic generation counter, no futex; calling thread participates | **6** | +69% decode（barrier 500× 更快） |
| **AVX2 SwiGLU** — Cephes exp2 polynomial, 8-wide SIMD sigmoid | **6** | 消除 ~4.9 ms/token scalar exp |
| **AVX2 RMSNorm** — vectorised sum-of-squares + scale | **6** | 消除标量 RMSNorm 循环 |
| **GEMV 内循环 _mm_prefetch** — Q4K outer block loop 预取下一块到 L1 (AVX2 + VNNI) | **7** | ~3–4% decode（减少 L3→L2 stall） |
| **Fusion A: rms_norm_out** — 双输入 RMSNorm 消除 E=2048 副本 | **7** | 减少每层两次 8KB memcpy |
| **Fusion C: gate+up+swiglu 单 parallel_for** — 22 层少 22 次 barrier | **7** | ~2% decode（减少 barrier + up[] L2 往返） |
| **Fusion E: rope_and_store_k** — RoPE K 直接写入 KV cache | **7** | 消除 KVH×HD float 每层中间 copy |
| **Q4K 8-row repack** (`block_q4_kx8`) — 行交织顺序布局消除 DDR4 行切换 | **8** | +5% decode 7B, +1–3% (1.1B) |
| **VNNI r8 kernel** — `dpbusd` 替换 `maddubs+madd`（INT8 dot 2× 吞吐量） | **9** | 与 Phase 8 AVX2 r8 持平（VNNI hw 已 DDR4 饱和） |
| **vector fnmadd min-correction** — `sum0f` 模式用 `_mm256_fnmadd_ps` 替代标量 hsum | **9** | 消除 `sum_q8` hsum 瓶颈，+1–2% |
| **18-cache-line prefetch** — r8 内核预取覆盖完整 1152 字节 r8 block | **9** | 减少 L3→L2 stall，补全预取窗口 |
| **j 默认化 16** — 匹配 i9-12900K P-core 逻辑 CPU 数 | **10** | +3–4 tok/s |
| **Per-CPU 线程绑定** — worker[i]→CPU i，主线程→CPU n_workers | **10** | 消除 E-core 迁移和同 CPU 竞争，减少抖动 |
| **静态权重缓存**（`WT::r8` 为 `shared_ptr`）— 同进程多次调用跳过 130 ms 重排 | **10** | TTFT 400 ms → 150 ms（同进程第二轮） |
| **静态 ThreadPool 缓存** — worker 永久自旋，再次唤醒 <200 ns | **10** | TTFT 省去 ~20 ms 线程池重建开销 |

---

## 6. Running the Benchmark

### Available GGUF Files

| Model | File | Size | Path | Status |
|-------|------|------|------|--------|
| TinyLlama-1.1B-Chat | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | ~670 MB | `models/tinyllama-1.1B/` | ✅ 可用 |
| LLaMA-2-7B | `llama-2-7b.Q4_K_M.gguf` | ~3.8 GB | `models/llama-2-7B/` | ✅ 可用 |
| Qwen2.5-0.5B-Instruct | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | ~491 MB | `models/qwen2.5-0.5B/` | ✅ smoke passed |
| DeepSeek-R1-Distill-Qwen-1.5B | `DeepSeek-R1-Distill-Qwen-1.5B.Q4_K_M.gguf` | ~1.12 GB | `models/deepseek-r1-distill-qwen-1.5B/` | ✅ smoke passed |
| Qwen2.5-3B-Instruct | `qwen2.5-3b-instruct-q4_k_m.gguf` | ~2.1 GB | `models/qwen2.5-3B/` | ⏳ 待下载/smoke |
| Qwen2.5-Coder-3B-Instruct | `qwen2.5-coder-3b-instruct-q4_k_m.gguf` | ~2.1 GB | `models/qwen2.5-coder-3B/` | ⏳ 待下载/smoke |
| Qwen3-4B | `Qwen3-4B-Q4_K_M.gguf` | ~2.5 GB | `models/qwen3-4B/` | 🧪 experimental |
| Hunyuan-1.8B-Instruct | `hunyuan-1.8b-instruct-q4_k_m.gguf` | ~1.13 GB | `models/hunyuan-1.8B/` | 🧪 experimental |
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

## 7. Phase 10 结果总结 & Phase 11 Roadmap

### Phase 10 已完成（TinyLlama-1.1B Q4_K_M）

| Feature | Status | Actual gain |
|---------|--------|-------------|
| j 默认化 14→16 | ✅ | +3–4 tok/s |
| VNNI r8 kernel（`gemv_q4k_r8_avxvnni_range`） | ✅ | 持平 Phase 8（VNNI hw DDR4 饱和） |
| vector fnmadd min-correction | ✅ | +1–2% |
| 18-cache-line prefetch | ✅ | ~0.5% |
| P-core 进程级 affinity + per-CPU 绑定 | ✅ | 减少抖动，均値 +1–2 tok/s |
| 静态权重缓存（shared_ptr r8） | ✅ | 多轮 TTFT 直接命中，省 130 ms |
| 静态 ThreadPool 缓存 | ✅ | 多轮省 ~20 ms |
| **总计** | — | **~69 → ~75 tok/s (+8.4%)** |

### Phase 11 Roadmap

| Feature | Expected gain | Status |
|---------|--------------|--------|
| 磁盘缓存 repack buffers（.axon 缓存文件） | TTFT 冷启动 ~150 ms | ⏳ |
| Prefill GEMM L2 blocking（T≥64 prompt） | ~2× prefill tok/s | ⏳ |
| Q6K AVX-VNNI path | ~5%（Q6K models） | ⏳ |
| 多轮 KV cache 持久化 API | 第二轮 TTFT 减少 KV alloc | ⏳ |
| CUDA backend (GTX 1080 Ti) | ~5–8× decode | ⏳ |
