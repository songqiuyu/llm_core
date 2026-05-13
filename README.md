# AxonForge (llm_core)

现代 C++20 端侧 LLM 推理引擎，x86 Linux 优先（AVX2 / FMA / F16C），支持 GPT-2、LLaMA-2、TinyLlama、Qwen2.5、DeepSeek-R1-Distill-Qwen 等模型。

---

## 构建

### 依赖

| 工具 | 版本要求 |
|------|----------|
| CMake | ≥ 3.16 |
| GCC / Clang | C++20 支持（GCC ≥ 11，Clang ≥ 13） |
| CPU | AVX2 + FMA + F16C（Intel Haswell / AMD Zen 1 及以上） |

### 编译步骤

```bash
# 配置（Release 模式，默认）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 多核编译
cmake --build build -j$(nproc)

# 运行单元测试（当前 49/49 通过）
ctest --test-dir build --output-on-failure
```

> Debug 构建：将 `Release` 换成 `Debug`，会关闭 `-O3` 并保留调试符号。

---

## 模型下载

> 所有模型放在项目根目录的 `models/` 下，需手动创建对应子目录。

### GPT-2 117M（F16 GGUF）

GPT-2 的原始权重以旧版 GGML 格式分发，需先下载再用仓库内的转换脚本生成 GGUF：

```bash
mkdir -p models/gpt-2-117M

# 1. 从 HuggingFace 下载 GGML 权重（~240 MB）
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='ggml-org/gpt-2',
                filename='ggml-model.bin',
                local_dir='models/gpt-2-117M')
"

# 2. 转换为 GGUF v3
python3 tools/convert_ggml_gpt2_to_gguf.py \
    --input  models/gpt-2-117M/ggml-model.bin \
    --output models/gpt-2-117M/gpt2-117M-f16.gguf
```

### TinyLlama-1.1B-Chat-v1.0（Q4_K_M GGUF）

```bash
mkdir -p models/tinyllama-1.1B

# 方式一：huggingface-cli（推荐）
pip install huggingface_hub
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --local-dir models/tinyllama-1.1B

# 方式二：wget 直接下载
wget -P models/tinyllama-1.1B \
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

### LLaMA-2-7B（Q4_K_M GGUF）

> **注意**：LLaMA-2 需要先在 [Meta 官网](https://ai.meta.com/llama/) 申请访问权限，并在 HuggingFace 上完成授权，才能下载。

```bash
mkdir -p models/llama-2-7B

# 授权后通过 huggingface-cli 下载（~3.8 GB）
huggingface-cli download TheBloke/Llama-2-7B-GGUF \
    llama-2-7b.Q4_K_M.gguf \
    --local-dir models/llama-2-7B
```

### Qwen2.5-0.5B-Instruct（Q4_K_M GGUF）

```bash
mkdir -p models/qwen2.5-0.5B

# 新版 huggingface_hub 使用 hf 命令；如果只有旧 huggingface-cli，请先升级：
# pip install -U huggingface_hub
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --local-dir models/qwen2.5-0.5B
```

### DeepSeek-R1-Distill-Qwen-1.5B（Q4_K_M GGUF）

```bash
mkdir -p models/deepseek-r1-distill-qwen-1.5B

hf download QuantFactory/DeepSeek-R1-Distill-Qwen-1.5B-GGUF \
    DeepSeek-R1-Distill-Qwen-1.5B.Q4_K_M.gguf \
    --local-dir models/deepseek-r1-distill-qwen-1.5B
```

---

## 快速上手

### GPT-2

```bash
build/tools/cli/axonforge-cli -m models/gpt-2-117M/gpt2-117M-f16.gguf -p "The capital of France is" -n 64
```

### LLaMA

```bash
build/tools/cli/axonforge-cli -m models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "The capital of France is" -n 64

build/tools/cli/axonforge-cli -m models/llama-2-7B/llama-2-7b.Q4_K_M.gguf -p "The capital of France is" -n 64
```

### Qwen / DeepSeek

```bash
build/tools/cli/axonforge-cli \
    -m models/qwen2.5-0.5B/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -p "The capital of France is" -n 64 -t 0.0

build/tools/cli/axonforge-cli \
    -m models/deepseek-r1-distill-qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B.Q4_K_M.gguf \
    -p "What is 2+2?" -n 64 -t 0.0
```

---

## CLI 使用说明

### 单次生成模式

```
axonforge-cli -m <model.gguf> -p <prompt> [选项]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `-m <path>` | — | GGUF 模型文件路径（必填） |
| `-p <text>` | — | 输入 prompt（单次模式必填） |
| `-n <int>` | 256 | 最大生成 token 数 |
| `-t <float>` | 0.8 | 温度（0.0 = greedy） |
| `--top-k <int>` | 40 | Top-K 采样 |
| `--top-p <float>` | 0.95 | Top-P 采样 |
| `-j <int>` | 16 | 推理线程数（含主线程；自动 `min(16, nproc)`） |
| `-V` / `--verbose` | off | 打印详细推理信息（见下文） |
| `-b <id>` | cpu_x86 | 后端选择 |

示例：

```bash
# greedy 解码，打印详细统计
build/tools/cli/axonforge-cli \
    -m models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -p "Once upon a time" -n 128 -t 0.0 -V
```

### 对话模式（多轮连续对话）

```
axonforge-cli -m <model.gguf> -i [选项]
```

启动后显示 AxonForge 字符画 Banner 及模型信息，然后进入交互提示符 `>>>`。

**对话内置命令：**

| 命令 | 说明 |
|------|------|
| `/help` | 显示命令帮助 |
| `/new` | **清空对话历史**，开始新话题 |
| `/clear` | 清屏 |
| `/quit` 或 `/exit` | 退出（也可 Ctrl-D） |

**历史记忆机制：** 每轮输入都会携带完整的历史上下文，使模型能够理解前文。上下文超过 4096 tokens 时自动从最旧的 token 开始截断。`-V` 模式下提示符显示当前 context 大小：`[ctx:247] >>>`。

示例：

```bash
# 开启详细统计的多轮对话
build/tools/cli/axonforge-cli \
    -m models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -i -n 200 -t 0.7 -V
```

### `-V` 详细推理统计

加上 `-V` 后，每次生成结束会输出：

```
┌─ Inference Stats ────────────────────────────
│  Context tokens  : 247
│  Generated       : 64
│  TTFT            :   884.0 ms    ← 首 token 延迟（含 prefill）
│  Prefill speed   :     7.90 tok/s ← 输入 token 处理速度
│  Decode speed    :    20.12 tok/s ← 生成速度
│  Total time      :  4681.1 ms
│  Memory (RSS)    :   613 MB
└────────────────────────────────────────────
```

不加 `-V` 时输出简化统计：`[stats] TTFT: 884 ms | 100 tokens | 20.12 tok/s`

---

## 性能基准（i9-12900K，DDR4-3200，Release 构建）

### 当前结果（Phase 10）

| 模型 | 精度 | 线程数 | decode tok/s | TTFT (warm) | vs llama.cpp |
|------|------|--------|-------------|-------------|-------------|
| GPT-2 117M | F16 | 16 | **~145** | ~50 ms | +190% |
| TinyLlama-1.1B | Q4_K_M | 16 | **~75** | ~285 ms | **~97%** |
| LLaMA-2-7B | Q4_K_M | 16 | **~11.5** | ~1.5 s | ~77% |
| Qwen2.5-0.5B-Instruct | Q4_K_M | 16 | **~30.8** | ~352 ms | ~2.9x ¹ |
| DeepSeek-R1-Distill-Qwen-1.5B | Q4_K_M | 16 | **~50.2** | ~552 ms | ~5.6x ¹ |

> **内存带宽利用率**（DDR4 理论峰值 51.2 GB/s）：TinyLlama Phase 10 约 73%（~37.5 GB/s），其中约 20% 为多线程调度/同步开销。
> ¹ 本机 `llama.cpp` b9093、CPU-only、`-t 16 -c 4096 --single-turn --no-warmup --perf` 的 smoke 对比；Qwen/DeepSeek 仍未做 logits/token 对齐，只作为端侧 smoke 性能参考。

### 历史进展

| Phase | 内容摘要 | TinyLlama decode | LLaMA-2-7B decode |
|-------|---------|-----------------|-------------------|
| Phase 3 | 基础 LLaMA Q4_K_M + batched prefill | ~20.1 tok/s | ~5.0 tok/s |
| Phase 4 | AVX2 attention + GEMM prefill | ~23.4 tok/s | ~6.1 tok/s |
| Phase 5 | AVX-VNNI Q4K 散射内核 | ~25.5 tok/s | ~6.4 tok/s |
| Phase 6 | Spinwait 线程池 + Release + AVX2 归一化 | ~65.7 tok/s | ~10.9 tok/s |
| Phase 7 | Fusion A/C/E + 预取 + 4-row tiling | ~68–70 tok/s | ~10.9 tok/s |
| Phase 8 | Q4K 8-row r8 行交织重排 (AVX2) | ~70–71 tok/s | ~11.5 tok/s |
| Phase 9 | VNNI r8 kernel + min-correction vector fnmadd + 18-CL 预取 | ~69–70 tok/s | ~11.5 tok/s |
| **Phase 10** | **j=16 + CPU 亲和性绑定 + 静态权重/线程池缓存** | **~75 tok/s** | ~11.5 tok/s |

---

## Phase 10 优化分析

### Phase 10 新增改进（相对 Phase 9 baseline ~69 tok/s → ~75 tok/s，+8.4%）

#### 1. 线程数默认值 14 → 16

i9-12900K 的 P-core（Performance Core）有 12 个物理核 × 2 超线程 = 24 个逻辑 CPU（编号 0–23），E-core 占 CPU 24–27。旧默认 `j=14`（13 workers）未充分使用 HT；改为 `j=16`（15 workers + 1 main）后，所有 8 个高频 P-core 物理核均被两个推理线程利用，内存带宽饱和度提升。

#### 2. P-core 进程级 CPU 亲和性 + 每线程 CPU 绑定

**问题**：OS 可能将工作线程迁移到 E-core（~1.5 GHz vs P-core 3.2–5.2 GHz），或将多个线程调度在同一个物理核上，导致 decode 速度偶发性从 75 tok/s 跌至 60 tok/s。

**设计（与 llama.cpp 的关键差异）**：

```
llama.cpp：依赖 OS 调度，无 CPU 绑定（可选 --cpu-mask 但默认关闭）
AxonForge：
  ① sched_setaffinity(0, {CPU 0..15})    — 进程级屏蔽 E-core
  ② pthread_setaffinity_np(worker_i, {CPU i})  — 每个 worker 独占一个逻辑 CPU
  ③ ThreadPool::pin_caller(n_workers)    — 主线程独占 CPU n_workers (=15)
```

效果：每个线程（包括主线程串行段）都在独占的 P-core 上运行，消除了 OS 调度引起的速度抖动。

#### 3. 静态权重缓存（`s_wt_cache` / `WT::r8` 改为 `shared_ptr`）

**问题**：每次 `llama_generate` 调用都重新执行 Q4K → r8 格式转换，需要申请 ~520 MB 内存并写满，触发 ~13 万次内核 page fault，约耗时 130 ms。

**设计**：
```cpp
// WT::r8 改为 shared_ptr，使 LlamaWeights 可 O(1) 拷贝
std::shared_ptr<std::vector<uint8_t>> r8;

// llama_generate 内静态缓存，同进程二次调用直接命中
static const Engine* s_wt_engine = nullptr;
static LlamaWeights  s_wt_cache;
if (s_wt_engine != &engine) {          // 仅首次 build
    s_wt_cache  = build_weights(...);
    s_wt_engine = &engine;
}
state.weights = s_wt_cache;            // O(1)：仅 shared_ptr refcount++
```

**效果（多轮对话模式）**：第 2 次及以后 `llama_generate` 调用跳过 130 ms 重排，TTFT 从 ~400 ms 降至 ~150 ms。

#### 4. 静态 ThreadPool 缓存

**问题**：每次调用创建 15 个 worker 线程（`pthread_create`）约需 20 ms，且每次等待 `live_count_` barrier。

**设计**：
```cpp
static std::unique_ptr<ThreadPool> s_pool;
static int s_pool_nw = 0;
if (!s_pool || s_pool_nw != n_workers) {
    s_pool    = std::make_unique<ThreadPool>(n_workers);
    s_pool_nw = n_workers;
}
```

Worker 线程在首次调用后永久自旋（`kSpinBudget=50000 × _mm_pause` ≈ 0.5 ms 后 yield），再次唤醒只需 ~200 ns。**相比 llama.cpp 每次上下文销毁重建线程池，AxonForge 的多轮对话不需要重启线程。**

---

## Phase 6 优化分析

### 根因定位

Phase 5 decode 速度仅 ~25 tok/s，但内核文件（`gemv_q4k_q8_avx2.cpp` 等）均已用 `-O3` 编译，问题集中在两处：

#### 1. 主体代码一直在 Debug 模式下编译（最大影响）

`CMakeLists.txt` 的 `-O3` 标志受 `$<$<CONFIG:Release>:...>` 保护，项目实际以 `CMAKE_BUILD_TYPE=Debug`（即 `-O0 -g`）运行。**GEMV 内核因单独设置了 `-O3` 而较快，但 forward pass 中所有其他代码——thread pool lambda、残差加法、Q8 量化、RoPE——全部是未优化的 Debug 代码。**

修复：始终用 `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` 配置。

#### 2. 线程池使用 mutex + condvar 导致高唤醒延迟（第二大影响）

旧线程池（`thread_pool.hpp`）的 `parallel_for` 设计：

```
主线程               工作线程 0~N-1
  │  acquire mtx         │
  │  update task state   │  ← sleeping in condvar.wait()
  │  notify_all ─────────►  wake up (futex syscall ~10-100 μs each)
  │                      │  acquire mtx → read task → do work
  │  condvar.wait ◄──────── pending-- → notify_done
  │  (blocked)           │
```

每次 `parallel_for` 调用需要工作线程从内核睡眠状态唤醒。在 Linux 上，`futex_wake` + 线程调度恢复约 **10–100 μs**。

每个 decode token 共调用 `parallel_for` **155 次**（22 层 × 7 次矩阵操作 + lm_head），总开销：
```
155 次 × 100 μs = ~15–20 ms/token（占单线程 81 ms 的 25%）
```

此外，**主线程不参与计算**，只是等待工作线程完成，浪费了一个核心。

#### 修复：Spinwait 线程池

新 `thread_pool.hpp` 核心改动：

```cpp
// 工作线程：自旋等待新任务（不进内核睡眠）
while ((g = generation_.load(acquire)) == last_gen) {
    _mm_pause();                      // ~10 ns 一次，不进入内核
    if (++spins >= 50000) {           // ~0.5 ms 后 yield 防止系统饥饿
        spins = 0;
        std::this_thread::yield();
    }
}

// parallel_for：主线程也做最后一个 chunk
generation_.fetch_add(1, release);   // 原子发布，无 mutex
fn(nw * chunk, min(N, (nw+1)*chunk)); // 主线程参与计算
while (pending_.load(acquire) > 0)   // 自旋等待，无 futex
    _mm_pause();
```

效果：
- barrier 开销从 ~100 μs → ~200 ns（500× 改进）
- 主线程贡献 `1/(N+1)` 的计算量（N=13 workers → 14 分之一额外吞吐）
- 155 次 barrier × 200 ns = 0.03 ms（vs 原来 ~20 ms）

#### 3. SwiGLU 中的标量 `std::exp`（中等影响）

每层 FFN 激活函数 `silu(x) * up[x]` 需要对 FD=5632 个元素调用 `std::exp(-x)`，每次 ~30 ns：

```
22 层 × 5632 × 30 ns = ~3.7 ms/token
```

修复：用 AVX2 Cephes 多项式替换，8 路 SIMD 并行计算 exp，同时向量化 sigmoid 和乘法：
```cpp
// exp_avx2_(): 度-5 Chebyshev 多项式，误差 ~2^-23，约 8 条 SIMD 指令
__m256 e = exp_avx2_(neg_abs_g);
```

#### 4. RMSNorm 标量循环（小影响）

每层 2 次 RMSNorm（44 次/token），每次两个 O(E) 循环：

```cpp
// 旧（标量）
for (int i = 0; i < n; i++) ss += x[i] * x[i];   // 无 SIMD
for (int i = 0; i < n; i++) x[i] = w[i] * x[i] * ss;

// 新（AVX2，8× 并行）
for (; i + 8 <= n; i += 8)
    acc = _mm256_fmadd_ps(v, v, acc);   // vfmadd256
```

### Amdahl 定律分析

用 Debug 构建的线程扩展数据估算：

| 参数 | 值 |
|------|-----|
| 单线程 T₁ | 81.7 ms/token |
| 4 线程 T₄ | 38.1 ms/token |
| 可并行部分（GEMV） | 71%（58 ms） |
| 固有串行部分 | **29%（23.6 ms）** |
| 理论极限（∞ 线程） | ~23.6 ms = 42 tok/s |

即使有无限线程，Debug 模式下最多只能达到 **42 tok/s**，因为串行开销（主要是 condvar 唤醒延迟）不随线程数缩放。Release 构建消除了非内核代码的冗余开销，spinwait pool 消除了同步延迟。

---

## 模型文件

| 模型 | 路径 |
|------|------|
| GPT-2 117M | `models/gpt-2-117M/gpt2-117M-f16.gguf` |
| TinyLlama-1.1B-Chat | `models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` |
| LLaMA-2-7B | `models/llama-2-7B/llama-2-7b.Q4_K_M.gguf` |
| Qwen2.5-0.5B-Instruct | `models/qwen2.5-0.5B/qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| DeepSeek-R1-Distill-Qwen-1.5B | `models/deepseek-r1-distill-qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B.Q4_K_M.gguf` |
