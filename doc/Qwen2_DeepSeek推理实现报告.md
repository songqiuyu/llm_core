# AxonForge Qwen2.5 / DeepSeek-R1-Distill-Qwen 推理实现报告

> 时间：2026-05-13  
> 目标模型：Qwen2.5-0.5B-Instruct Q4_K_M、DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M  
> 代码：`include/axonforge/models/qwen2.hpp`、`src/models/qwen2_model.cpp`、`src/models/llama_model.cpp`

---

## 一、目标与范围

本阶段目标是让端侧 x86 CPU 路径支持小型 Qwen 系 GGUF 模型端到端推理：

- `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- `QuantFactory/DeepSeek-R1-Distill-Qwen-1.5B-GGUF`

范围限定为 dense Qwen2/Qwen2.5、DeepSeek-R1-Distill-Qwen，以及实验性的 dense Qwen3。暂不支持 Qwen2-MoE、Qwen3-MoE、Qwen3-Next、Qwen3-VL、DeepSeek-V2/V3/R1 原版 MoE/MLA。

---

## 二、架构差异

Qwen2 与现有 LLaMA-family 路径非常接近：

| 模块 | LLaMA/TinyLlama | Qwen2.5 / Distill-Qwen |
|------|-----------------|------------------------|
| Norm | RMSNorm | RMSNorm |
| Position | RoPE | RoPE |
| FFN | SwiGLU | SwiGLU |
| Attention | MHA/GQA | GQA |
| 权重格式 | F16/Q4_K_M/Q6_K | F16/Q4_K_M/Q6_K |
| 额外差异 | QKV 通常无 bias | Q/K/V projection 带 bias |
| Tokenizer | SentencePiece/简化 longest-match | byte-level BPE + Qwen2 pre-tokenizer |

因此实现采用独立 `qwen2_model.cpp` 对外暴露 Qwen2 API，但生成阶段复用现有 LLaMA-like transformer forward 和 x86 kernel。共享 forward 已补充 Q/K/V bias 支持；LLaMA 模型无 bias 时该分支为空操作。

Qwen3 dense 模型继续沿用 LLaMA-like forward，但相比 Qwen2 额外包含每层 `blk.N.attn_q_norm.weight` 和 `blk.N.attn_k_norm.weight`。当前实现把这两个 RMSNorm 做成可选 tensor：存在时在 Q/K projection 之后、RoPE 之前按 head_dim 独立归一化；不存在时保持 Qwen2/LLaMA 原路径。

---

## 三、实现内容

### 3.1 Qwen2 API 与路由

新增：

- `Qwen2Config`
- `qwen2_generate()`
- `qwen2_encode()`
- `qwen2_decode()`

`Engine::encode/decode` 和 `Session::generate` 在 `general.architecture == "qwen2"` 或 `"qwen3"` 时分发到 Qwen 路径。CLI 也显式识别 `qwen2/qwen3`，保持 streaming callback 与统计输出一致。

### 3.2 QKV bias

`LlamaWeights` 的 layer descriptor 增加：

- `blk.N.attn_q.bias`
- `blk.N.attn_k.bias`
- `blk.N.attn_v.bias`

加载时这些 tensor 可选；若存在，则在 Q/K/V projection 后逐元素加 bias。batched prefill 路径也对每个 token row 加同一组 bias。

### 3.3 Tokenizer metadata

`Engine` 现在保存并暴露：

- `tokenizer.ggml.tokens`
- `tokenizer.ggml.merges`
- `tokenizer.ggml.token_type`
- `tokenizer.ggml.pre`

Qwen2 tokenizer 使用 merge-rank BPE。`raw=true` 时会先解析 control/user-defined special tokens，供 chat template prompt 使用；decode 阶段会过滤 control token，避免 `<|im_start|>`、`<|im_end|>` 进入正文。

Qwen3-4B 的 GGUF tokenizer 继续使用 Qwen BPE/ChatML 路线；`/think`、`/no_think` 作为普通 prompt 内容进入模型，由模型内 chat/template 约定处理。

### 3.4 复用的性能路径

Qwen2 generation 复用：

- Q4_K_M / Q6_K / F16 dtype dispatch
- GGUF Q5_0 / Q8_0 权重解码 fallback
- Q8_0 weight × Q8_0 activation AVX2 int8 dot
- Q8_0 activation quantization
- AVX2 / AVX-VNNI GEMV
- Q4K 8-row r8 repack
- spinwait ThreadPool
- P-core affinity
- batched prefill
- top-k / top-p / repetition penalty sampling

默认上下文 cap 仍为 4096，避免 Qwen2.5 32K context 在端侧直接分配过大 KV cache。

---

## 四、验证

已完成：

```bash
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

当前单元测试：54/54 通过。

新增测试覆盖：

- `arch=qwen2` metadata 解析
- Qwen2 layer/head/kv-head/rope/rms/vocab 配置
- Qwen2.5-3B/Coder-3B 风格 metadata 解析
- `arch=qwen3` metadata、ChatML 模板和 Qwen BPE 路由
- BPE merges 基本合并
- raw special token 解析与 decode control token 过滤

端到端 smoke 命令：

```bash
build/tools/cli/axonforge-cli \
    -m models/qwen2.5-0.5B/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -p "The capital of France is" -n 16 -t 0.0

build/tools/cli/axonforge-cli \
    -m models/deepseek-r1-distill-qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B.Q4_K_M.gguf \
    -p "What is 2+2?" -n 32 -t 0.0
```

Smoke 结果：

- Qwen2.5-0.5B：输出包含 Paris/France，64 token，decode 约 30.8 tok/s，TTFT 约 352 ms，RSS 约 450 MB。
- DeepSeek-R1-Distill-Qwen-1.5B：输出 “four”，32 token，decode 约 50.2 tok/s，TTFT 约 552 ms，RSS 约 1.57 GB。

---

## 五、已知限制

- Qwen2 tokenizer 的 regex pre-tokenizer 覆盖常见英文、数字、空白、标点和 UTF-8 文本；后续若要 logits/token IDs 精确对齐 llama.cpp，需要增加随机 tokenizer 回归测试。
- 当前不做 llama.cpp logits 对齐，不做性能对标。
- Qwen3 为 dense 4B 实验支持，已接入 Q/K head-wise RMSNorm；仍需真实模型 smoke 与 tokenizer 对齐测试后再视为稳定支持。
- Qwen2 r8 repack 是速度优先策略；DeepSeek 会额外占用约 500-600 MB repack 内存。若端侧设备更看重低内存，可关闭 Qwen2 r8 并回到 F32 activation fallback。
- Q5_0 目前仍使用 F32 activation fallback；尝试过标量 Q5_0×Q8 路径，速度更慢，后续需要 SIMD unpack 后再接入。
- `output.bias` 尚未接入；目标 Qwen2.5-0.5B GGUF 为 tied embeddings，第一阶段不依赖该 bias。
- 不支持 Qwen3-MoE、Qwen3-Next、Qwen3-VL 和 DeepSeek 原版 MoE/MLA。
