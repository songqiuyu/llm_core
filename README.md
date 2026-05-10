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
│  TTFT            :   334.0 ms    ← 首 token 延迟（含 prefill）
│  Prefill speed   :   739.0 tok/s ← 输入 token 处理速度
│  Decode speed    :    14.80 tok/s ← 生成速度
│  Total time      :  4681.1 ms
│  Memory (RSS)    :   613 MB
└────────────────────────────────────────────
```

不加 `-V` 时输出简化统计：`[stats] TTFT: 334 ms | 64 tokens | 14.80 tok/s`

---

## 性能基准（i9-12900K，8 线程，AVX2）

| 模型 | 精度 | tok/s | vs 标量基线 |
|------|------|-------|------------|
| GPT-2 117M | F16 | **73.2** | — |
| TinyLlama-1.1B | Q4_K_M | **~14.8** | 15.4× |
| LLaMA-2-7B | Q4_K_M | **~1.79** | 9.9× |

---

## 模型文件

| 模型 | 路径 |
|------|------|
| GPT-2 117M | `models/gpt-2-117M/gpt2-117M-f16.gguf` |
| TinyLlama-1.1B-Chat | `models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` |
| LLaMA-2-7B | `models/llama-2-7B/llama-2-7b.Q4_K_M.gguf` |