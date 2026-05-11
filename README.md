# AxonForge (llm_core)

现代 C++20 端侧 LLM 推理引擎，x86 Linux 优先（AVX2 / FMA / F16C），支持 GPT-2、LLaMA-2、TinyLlama 等模型。

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

# 运行单元测试（当前 47/47 通过）
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
| `-j <int>` | 14 | 推理线程数（含主线程；自动 `min(14, nproc)`） |
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

### 当前结果（Phase 6）

| 模型 | 精度 | 线程数 | decode tok/s | TTFT (warm) | vs Phase 5 | vs llama.cpp |
|------|------|--------|-------------|-------------|------------|-------------|
| GPT-2 117M | F16 | 14 | **~145** | ~50 ms | — | — |
| TinyLlama-1.1B | Q4_K_M | 14 | **~65.7** | ~100 ms | **+158%** | ~95% |
| LLaMA-2-7B | Q4_K_M | 14 | **~10.9** | ~1.5 s | **+70%** | ~73% |

### 历史进展

| Phase | 内容 | TinyLlama decode | LLaMA-2-7B decode |
|-------|------|-----------------|-------------------|
| Phase 3 | 基础 LLaMA Q4_K_M | ~20.1 tok/s | ~5.0 tok/s |
| Phase 4 | AVX2 attention + GEMM prefill | ~23.4 tok/s | ~6.1 tok/s |
| Phase 5 | AVX-VNNI Q4K 内核 | ~25.5 tok/s | ~6.4 tok/s |
| **Phase 6** | **线程池重写 + Release 构建 + AVX2 归一化** | **~65.7 tok/s** | **~10.9 tok/s** |

> **内存带宽利用率**（DDR4 理论峰值 51.2 GB/s）：TinyLlama Phase 6 约 87%（~44.6 GB/s），已接近 DDR4 带宽上限。

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