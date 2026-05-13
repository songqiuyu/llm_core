# AxonForge Phase 9/10 优化详解 & 与 llama.cpp 差异分析

> 时间：2026-05-12  
> 硬件：Intel i9-12900K (8P-core×2HT=16 logical P-cores, 4 E-cores), DDR4-3200 51.2 GB/s  
> 模型：TinyLlama-1.1B Q4_K_M（514 MB 权重带宽/token）  
> 基准：Phase 9 ~69 tok/s → Phase 10 **~75 tok/s** (+8.4%)

---

## 一、性能上限分析

| 指标 | 值 |
|------|-----|
| DDR4-3200 双通道理论带宽 | 51.2 GB/s |
| TinyLlama-1.1B 单 token 权重读取量 | ~514 MB（Q4K r8 格式） |
| 带宽上限 tok/s | 51.2 GB/s ÷ 514 MB ≈ **99.6 tok/s** |
| Phase 10 实测 | **~75 tok/s（75% 利用率）** |
| Phase 10 绝对上限缺口 | ~25%（线程调度+同步+串行段+DRAM 实际效率） |

DDR4 实际效率约 90–95%，串行段（RMSNorm、Q8Quant、attention、残差加法）占 ~6%，线程同步 ~2%。实测已接近合理极限。

---

## 二、Phase 9 优化详解

### 2.1 Q4K 8-row r8 内核原理（Phase 8 基础）

**文件**：`src/backend/cpu_x86/kernels/gemv_q4k_r8_avx2.cpp`

**原始 Q4K scatter 布局问题**：
```
传统 Q4K 单行布局（144 B/行）：
行0: [d₀][dmin₀][scales₀×12][qs₀×128]   144 B
行1: [d₁][dmin₁][scales₁×12][qs₁×128]   144 B  ← 非连续，行间距 = 整行大小
...
行7: [d₇][dmin₇][scales₇×12][qs₇×128]   144 B

访问 8 行时：8次独立 DDR4 行激活 → 行缓冲 thrash
```

**r8 交织布局**：
```
r8 块（1152 B = 8 行 × 144 B）：
[d₀][dmin₀][scales₀×12][qs₀×128]   ← 行0完整数据
[d₁][dmin₁][scales₁×12][qs₁×128]   ← 行1完整数据
...
[d₇][dmin₇][scales₇×12][qs₇×128]   ← 行7完整数据
1152 B 连续 → 1次 DDR4 行激活，硬件预取器可处理
```

**重排函数**（`repack_q4k_8rows`）：
```cpp
for (int g = 0; g < nrows/8; g++)
    for (int r = 0; r < 8; r++)
        copy_row(dst + g*1152 + r*144,   // 目标：r8 块内第r行
                 src + (g*8+r)*row_bytes); // 源：scatter格式原始行
```

---

### 2.2 AVX-VNNI r8 内核（Phase 9）

**文件**：`src/backend/cpu_x86/kernels/gemv_q4k_r8_avxvnni.cpp`  
**编译选项**：`-mavxvnni`（256-bit VNNI，不需要 AVX-512）

#### 2.2.1 dpbusd 替代 maddubs+madd

| 方案 | 指令数/32元素 | 延迟 | 吞吐 |
|------|-------------|------|------|
| llama.cpp scatter (maddubs+madd) | 2 | 5 cy | 0.5 ops/cy |
| AxonForge r8 AVX2 (maddubs+madd) | 2 | 5 cy | 0.5 ops/cy |
| **AxonForge r8 VNNI (dpbusd)** | **1** | **4 cy** | **1 ops/cy** |

```cpp
// AVX2 路径（2条指令）：
__m256i t = _mm256_maddubs_epi16(lo_u8, q8_i8);   // uint8×int8→int16
acc  = _mm256_add_epi32(acc, _mm256_madd_epi16(t, ones));  // int16→int32 累加

// VNNI 路径（1条指令）：
acc  = _mm256_dpbusd_epi32(acc, lo_u8, q8_u8);   // uint8×int8→int32 直接累加
```

> **注意**：在 TinyLlama-1.1B 上 VNNI r8 与 AVX2 r8 持平（~70 tok/s），因为 DDR4-3200 已接近饱和。计算及益被带宽掩盖。在更大模型（7B+）或更宽内存带宽（DDR5/LPDDR5）的机器上，VNNI 优势会更明显。

#### 2.2.2 vector fnmadd min-correction（关键差异）

Q4K 反量化公式：
```
weight[i] = d × scale[g] × nibble[i] - dmin × min_scale[g] × 1
```
其中 `-dmin × min_scale[g]` 是 min-bias 修正项，需要对 Q8_0 input 求和 `sum_q8 = Σxᵢ`。

**llama.cpp 风格**（scalar hsum 瓶颈）：
```cpp
// 对每个 Q8_0 block 进行横向求和
int32_t sum_q8 = 0;
for (int i = 0; i < 32; i++) sum_q8 += q8_data[i];   // 标量或 hsum
// 再乘以修正系数
acc -= dmin * min_scale * sum_q8;  // 串行依赖
```

**AxonForge Phase 9（向量化 fnmadd）**：
```cpp
// sum_q8 保持为 __m256i，延迟 cvt 到 fp32
__m256i q8_sum_vec = _mm256_add_epi32(
    _mm256_madd_epi16(_mm256_maddubs_epi16(q8_abs, ones), ones_16),
    existing_sum);
// 只在累加结束后一次性转换并修正：
__m256 sum0f = _mm256_cvtepi32_ps(q8_sum_vec);
acc_f = _mm256_fnmadd_ps(ms_vec, sum0f, acc_f);
// 等价于：acc_f -= ms_vec × sum0f
// 对 8 行同时操作，只有 1 次横向 hsum（而非 8 次）
```

**效果**：消除每个 Q8_0 block 的 ~5 cycle hsum 串行依赖，+1–2% decode。

#### 2.2.3 18-cache-line 全块预取

```cpp
// 旧（4 cache lines = 256 B，仅覆盖前 22%）：
for (int cl = 0; cl < 4; cl++)
    _mm_prefetch(next_blk + cl*64, _MM_HINT_T1);

// 新（18 cache lines = 1152 B，覆盖完整 r8 块）：
const char* nb = reinterpret_cast<const char*>(next_blk);
for (int cl = 0; cl < 18; cl++)
    _mm_prefetch(nb + cl*64, _MM_HINT_T1);
```

对 Alder Lake P-core L2 prefetch 队列（≥12 条流），18 个 `_mm_prefetch` 均在同一块内，不会超出硬件队列。覆盖整块确保下一块数据在 L2 就绪时不会出现 L3→L2 stall。

---

## 三、Phase 10 优化详解

### 3.1 j 默认值 14 → 16

**硬件拓扑**（i9-12900K）：
```
CPU 0–1:   P-core 0 (HT pair)
CPU 2–3:   P-core 1 (HT pair)
...
CPU 14–15: P-core 7 (HT pair)
CPU 16–23: P-core 8–11（未使用，j=16 时空闲）
CPU 24–27: E-core 0–3（排除在外）
```

j=14 时：CPU 14–15（P-core 7）空闲。j=16 后 P-core 7 的两个 HT 分别由 Worker 14 和调用线程占用，总吞吐提升。

**带宽模型**（近似）：
```
DDR4 实效带宽 ≈ 40 GB/s（测量值）
j=14: 14 流 × 2.86 GB/s/流 → 内存控制器未满
j=16: 16 流 × 2.5 GB/s/流 → 更多并发请求，DRAM bank 并行度更高
```

---

### 3.2 CPU 亲和性绑定设计

#### 与 llama.cpp 的对比

| 维度 | llama.cpp | AxonForge |
|------|-----------|-----------|
| 默认 CPU affinity | 无约束（OS 自由调度） | 进程级 `sched_setaffinity({0..n_threads-1})`，屏蔽 E-core |
| 线程迁移 | OS 可将线程从 P-core 迁到 E-core | 不可能（硬 affinity mask） |
| worker 绑定 | 无（或通过 --cpu-mask 全局设置） | `worker[i] → CPU i`（独占，`pthread_setaffinity_np`） |
| 调用线程绑定 | 无 | `pin_caller(n_workers)` → CPU 15（与所有 worker 无共享） |
| 线程间 CPU 竞争 | 可能（OS 超额调度） | 不可能（16 线程 × 16 CPU，1:1） |
| 实现代码 | N/A | `thread_pool.hpp`, `llama_model.cpp` |

#### 实现代码（`llama_model.cpp`）

```cpp
// ① 进程级 affinity：限制到 P-core 范围
cpu_set_t orig_aff, infer_aff;
sched_getaffinity(0, sizeof(orig_aff), &orig_aff);   // 保存原始
CPU_ZERO(&infer_aff);
for (int i = 0; i < n_threads; i++) CPU_SET(i, &infer_aff);
sched_setaffinity(0, sizeof(infer_aff), &infer_aff);  // 仅 CPU 0..15

// ② 静态 ThreadPool：workers 在此继承 {0..15} affinity，再立即 pin 到具体 CPU
if (!s_pool || s_pool_nw != n_workers)
    s_pool = std::make_unique<ThreadPool>(n_workers);  // worker[i] → CPU i

// ③ 调用线程绑定到 CPU n_workers（独占，无共享）
ThreadPool::pin_caller(n_workers);   // CPU 15

// ... 推理 ...

// ④ 推理结束后恢复原始 affinity
sched_setaffinity(0, sizeof(orig_aff), &orig_aff);
```

#### 实现代码（`thread_pool.hpp`）

```cpp
explicit ThreadPool(int n_workers) {
    workers_.reserve(n_workers);
    for (int i = 0; i < n_workers; i++)
        workers_.emplace_back([this, i] {
            pin_to_cpu_(i);       // worker i 独占 CPU i
            worker_loop(i);
        });
    // 等待所有 worker 自旋就绪后才返回
    while (live_count_.load(std::memory_order_acquire) < n_workers)
        _mm_pause();
}

static void pin_caller(int cpu_id) noexcept { pin_to_cpu_(cpu_id); }

static void pin_to_cpu_(int cpu_id) noexcept {
#ifdef __linux__
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(cpu_id, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#endif
}
```

#### 为什么之前的尝试失败

**失败方案**（per-worker pin，无 caller pin）：
```
worker[0] → CPU 0, worker[1] → CPU 1, ..., worker[13] → CPU 13
调用线程：未绑定 → OS 可调度到 CPU 14–27（包括 E-core!）
```
调用线程落在 E-core 后，串行段（RMSNorm、attention、残差加法等）慢 ~3×，导致整体 decode 从 73 降至 ~62 tok/s。

**正确方案**（本实现）：
```
worker[0..14] → CPU 0..14（各自独占）
调用线程     → CPU 15（独占，P-core 7 的第二个 HT）
```
全部 16 个线程在 P-core 上，无 E-core 迁移，无 CPU 间竞争。

---

### 3.3 静态权重缓存

#### 问题：每次调用重新 repack 的开销

Q4K → r8 转换过程：
1. `std::make_shared<std::vector<uint8_t>>(520MB)` → `malloc(520MB)` + 零初始化 → **~13万次 page fault × 0.5µs = ~65ms**
2. 逐行重排数据（读 514MB + 写 514MB @ 20 GB/s = ~51ms）
3. **合计：~116–130ms**

#### 解决方案

**步骤一**：`WT::r8` 类型改为 `shared_ptr`，使 `LlamaWeights` 可 O(1) 复制：
```cpp
struct WT {
    const void* data{nullptr};
    DType dtype{DType::UNKNOWN};
    // 旧：std::vector<uint8_t> r8;   ← 深拷贝，O(520MB) 时间
    // 新：
    std::shared_ptr<std::vector<uint8_t>> r8;  // O(1) refcount++
};
```

**步骤二**：静态缓存，同进程只 build 一次：
```cpp
static const Engine* s_wt_engine = nullptr;
static LlamaWeights  s_wt_cache;

if (s_wt_engine != &engine) {      // 首次 or 模型更换
    s_wt_cache  = build_weights(engine, L, E, H, KVH, FD);
    s_wt_engine = &engine;
}
state.weights = s_wt_cache;        // O(1)：仅 shared_ptr refcount bump
```

**效果**：
| 场景 | 旧行为 | 新行为 |
|------|--------|--------|
| CLI 单次调用 | 每次 repack ~130ms | 每次 repack ~130ms（无变化） |
| 交互模式第 2 轮 | 每次 repack ~130ms | 0ms（缓存命中） |
| TTFT 第 2 轮 | ~400ms | **~150ms** |

---

### 3.4 静态 ThreadPool 缓存

#### 问题：每次调用重建线程池的开销

```
pthread_create × 15 workers ≈ 1.5ms × 15 = ~22ms
live_count_ barrier（等待所有 worker spinning）≈ ~5ms
总计：~27ms/call
```

#### 解决方案

```cpp
static std::unique_ptr<ThreadPool> s_pool;
static int s_pool_nw = 0;

if (!s_pool || s_pool_nw != n_workers) {
    s_pool    = std::make_unique<ThreadPool>(n_workers);
    s_pool_nw = n_workers;
}
```

Worker 线程行为（`thread_pool.hpp`）：
```cpp
void worker_loop(int id) {
    live_count_.fetch_add(1, std::memory_order_release);  // 通知就绪
    int last_gen = 0;
    for (;;) {
        // Phase 1：自旋等待新任务（~50000 次 _mm_pause ≈ 0.5ms）
        while ((g = generation_.load(acquire)) == last_gen) {
            if (stop_) return;
            _mm_pause();
            if (++spins >= 50000) { spins = 0; std::this_thread::yield(); }
        }
        // Phase 2：执行任务
        last_gen = g;
        if (s < e) task_fn_(s, e);
        pending_.fetch_sub(1, release);
    }
}
```

唤醒延迟：`generation_.fetch_add(1, release)` → worker 在 `_mm_pause()` 循环中检测到变化 ≈ **~200 ns**（vs condvar 唤醒 ~10–100µs）。

#### 与 llama.cpp 线程池的对比

| 维度 | llama.cpp (`ggml_threadpool`) | AxonForge `ThreadPool` |
|------|------------------------------|------------------------|
| 同步机制 | pthread mutex + condvar（可配置为 spin） | **原子 generation 计数器** |
| 睡眠/唤醒开销 | ~10–100 µs/次（futex syscall） | **~200 ns/次（无系统调用）** |
| 调用线程参与计算 | 是（ggml_graph_compute 中） | **是**（做最后 1/(n+1) 份工作） |
| 线程池生命周期 | 随 `ggml_backend_cpu_set_n_threads` 创建 | **静态单例，进程内永久存活** |
| barrier 实现 | `atomic_int pending_tasks` + condvar | **`pending_.fetch_sub` + spinwait** |
| CPU 绑定 | `--cpu-mask` 进程级（无 per-thread） | **per-thread `pthread_setaffinity_np`** |
| 线程间 CPU 共享 | 可能（OS 超调度） | **不可能（1:1 绑定）** |

---

## 四、完整优化对比：AxonForge vs llama.cpp

### 4.1 内核层对比

| 内核 | llama.cpp | AxonForge | 差异 |
|------|-----------|-----------|------|
| Q4K GEMV (AVX2) | 4-row scatter，2 acc | 4-row scatter，2 acc | 基本相同 |
| Q4K GEMV (VNNI) | `dpbusd` scatter path | `dpbusd` **r8 顺序 8-row** | r8 减少 DDR4 行切换 |
| min-correction | 标量/向量 hsum 后乘 | **`_mm256_fnmadd_ps` 向量修正** | 消除 hsum 瓶颈 |
| prefetch | 4–8 cache lines | **18 cache lines（全块覆盖）** | 减少 L3→L2 stall |
| Q8_0 量化 | AVX2 向量化 | AVX2 向量化（相同） | 相同 |
| RMSNorm | 向量化 | 向量化，**`rms_norm_out` 消除中间 copy** | AxonForge Fusion A |
| SwiGLU | 标量 `expf` or AVX | **Cephes 多项式 + AVX2（8-wide）** | ~5× 加速 |
| Attention | 标量/AVX | **AVX2 dot + SAXPY** | ~5% 加速 |

### 4.2 内存管理对比

| 方面 | llama.cpp | AxonForge |
|------|-----------|-----------|
| 权重格式存储 | 原始 GGUF Q4K（scatter） | **r8 行交织重排（1152B/块）** |
| repack 时机 | context 创建时（llama_new_context）| `llama_generate` 首次调用 |
| repack 缓存 | 随 context 生命周期 | **静态 `s_wt_cache`，进程级永久** |
| 跨调用复用 | 依赖同一 context 对象 | **自动跨调用复用（同进程）** |
| 权重对象拷贝 | N/A（直接引用） | **`shared_ptr` O(1) 拷贝** |
| KV cache 分配 | context 创建时一次分配 | 每次 `llama_generate` 重新分配 |

### 4.3 并发模型对比

| 方面 | llama.cpp | AxonForge |
|------|-----------|-----------|
| 并行化粒度 | tensor 行分片 | **r8 组（8行）分片** |
| 同步 barrier | condvar 或 spin（可配置） | **原子 generation + spinwait** |
| barrier 开销 | ~10–100 µs（默认 condvar） | **~200 ns** |
| 调用线程角色 | 等待或协作（取决于配置） | **始终协作（1/n 份工作）** |
| CPU affinity | 进程级 mask（可选） | **per-thread pin + 进程级 mask** |
| 线程持久化 | context 生命周期内持久 | **进程内永久，静态单例** |

### 4.4 性能结果汇总

测试配置：TinyLlama-1.1B Q4_K_M，i9-12900K，DDR4-3200，Release 构建。

| 实现 | decode tok/s | vs llama.cpp |
|------|-------------|-------------|
| llama.cpp (j=8, 默认) | ~77 tok/s | — |
| llama.cpp (j=16) | ~77 tok/s（线程墙 ~8T） | — |
| AxonForge Phase 6 (j=14) | ~65.7 tok/s | 85% |
| AxonForge Phase 8 (j=14) | ~70–71 tok/s | 91% |
| AxonForge Phase 9 (j=14) | ~69–70 tok/s | 90% |
| **AxonForge Phase 10 (j=16)** | **~75 tok/s** | **~97%** |

> **llama.cpp 在 j>8 后 decode 速度不再提升**的原因：其线程池在内存饱和场景下调度不够高效，且未针对 P-core/E-core 混合架构做 affinity 约束。AxonForge 通过精确的 per-CPU 绑定在 j=16 时仍能有效利用所有 P-core HT 资源。

---

## 五、已验证失败的方案（负面清单）

| 方案 | 预期 | 实际 | 原因 |
|------|------|------|------|
| 1536-byte r8 block（预计算 FP32 scale） | -15%（减少 scale decode） | **-7%**（带宽 +33%） | DDR4 饱和场景下带宽增加不可接受 |
| lm_head (output.weight) r8 repack | +2% decode | **TTFT +35ms，decode ±0** | lm_head 只调用 1 次/token，r8 repack 无 8-row 分组优势；repack 本身耗时 |
| per-thread pin 无 caller pin | 消除 E-core 抖动 | **decode 62→68 tok/s 回退** | 调用线程落在 E-core，串行段 3× 变慢 |
| j=20（超过 P-core 数） | 更多并行 | **边际递减，部分 CPU 竞争** | DDR4 已饱和，超额线程只增加同步开销 |

---

## 六、残留优化空间（Phase 11+）

### 6.1 磁盘缓存 repack buffers（预计 -130ms 冷启动 TTFT）

```
首次运行：build_weights → 保存 r8 数据到 ~/.cache/axonforge/<model_hash>.r8
后续运行：直接 mmap r8 文件，跳过 repack
```

节省每次冷启动的 ~130ms（page fault + memcpy）。

### 6.2 Prefill GEMM L2 Blocking（T≥64，预计 2× prefill）

当前 GEMM 对每个输出行读取 T 个输入向量，当 T=64 时，输入 XQ8（64×64×36B=147KB）无法全部驻留 L2（1.25MB 够，但跨行分片后每线程仅 ~9KB）。L2 blocking 将输出行分块，保证同一块内的权重和输入均在 L2 内复用。

### 6.3 多轮 KV Cache 持久化

```cpp
// 当前：每次 llama_generate 分配 ~92MB KV cache（~6ms 零初始化）
static LlamaState s_state;  // 持久化，只在 nc/ne 变化时重分配
// 新对话：仅重置 n_past=0，不清空 KV 缓冲区（会被顺序覆写）
```

节省每次调用 ~6ms，并支持真正的多轮 KV cache 延续（无需重新处理历史 context）。

### 6.4 CUDA Backend（GTX 1080 Ti）

理论带宽 484 GB/s，预计 decode ~300–500 tok/s。需实现 `CudaTensorStorage`、权重 H2D 传输、`cudagemv_q4k` kernel，并通过现有 `IBackend::get_kernel()` 接口路由。
