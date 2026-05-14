# AxonForge Hunyuan Dense 推理调研与实现报告

> 时间：2026-05-13  
> 目标模型：Hunyuan-1.8B-Instruct Q4_K_M  
> 范围：腾讯混元 dense 文本模型；不包含 A13B/Large MoE、VL/OCR/Video/Image。

## 一、接入判断

混元 dense 系列包含 0.5B、1.8B、4B、7B，GGUF arch 为 `hunyuan-dense`。相比 A13B/Large MoE，它更适合当前端侧 CPU 路线：权重规模可控、仍是 dense decoder-only 主干，并能复用现有 Q4_K_M/Q6_K/F16/Q8_0 kernel。

第一阶段选择 `Hunyuan-1.8B-Instruct`，Q4_K_M 约 1.13GB，端侧内存压力接近 DeepSeek-R1-Distill-Qwen-1.5B。

## 二、架构差异

| 模块 | Qwen3 dense | Hunyuan dense |
|------|-------------|---------------|
| Norm | RMSNorm | RMSNorm |
| Position | RoPE | RoPE + XDRoPE alpha |
| Attention | GQA + Q/K norm | GQA + Q/K norm |
| Q/K norm 顺序 | RoPE 前 | RoPE 后 |
| FFN | SwiGLU | SwiGLU |
| Tokenizer | Qwen BPE | Hunyuan dense BPE |
| Chat | ChatML | `<｜hy_User｜>` / `<｜hy_Assistant｜>` |

实现上继续复用 LLaMA-like dense forward，但新增 `QkNormOrder`：Qwen3 使用 `BeforeRope`，Hunyuan dense 使用 `AfterRope`，其它 LLaMA/Qwen2 模型保持无 Q/K norm。

## 三、实现内容

- 新增 `HunyuanDenseConfig`、`hunyuan_dense_generate()`、`hunyuan_dense_encode()`、`hunyuan_dense_decode()`。
- `Engine` 支持 `general.architecture == "hunyuan-dense"` 的 encode/decode/generate 分发。
- `Engine::from_gguf()` 读取 `hunyuan-dense.rope.scaling.alpha`，并按 llama.cpp 规则调整 RoPE base：`theta * alpha^(head_dim/(head_dim-2))`。
- tokenizer 新增 `hunyuan-dense` pre-tokenizer：数字 1-3 位分组、中文/日文连续分组、通用字母/标点/空白分组。
- CLI `--chat` 对 `hunyuan-dense` 使用混元 dense fallback 模板。

## 四、验证

已完成：

```bash
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

当前单元测试：54/54 通过。

新增测试覆盖：

- `arch=hunyuan-dense` metadata 解析
- 1.8B 风格 layer/head/kv-head/ffn/context/vocab
- RoPE scaling alpha 修正
- Hunyuan dense tokenizer 的英文、数字分组、换行、中文 byte-level decode
- 混元特殊 token raw encode 与 decode 过滤

待真实模型下载后执行：

```bash
mkdir -p models/hunyuan-1.8B
hf download Edge-Quant/Hunyuan-1.8B-Instruct-Q4_K_M-GGUF \
    hunyuan-1.8b-instruct-q4_k_m.gguf \
    --local-dir models/hunyuan-1.8B

build/tools/cli/axonforge-cli \
    -m models/hunyuan-1.8B/hunyuan-1.8b-instruct-q4_k_m.gguf \
    -p "/no_think What is the capital of France?" -n 64 -t 0.0 -V
```

## 五、已知限制

- Hunyuan dense tokenizer 是按 llama.cpp 规则实现的轻量等价路径；后续如需 token ID 严格对齐，需要加入随机文本 tokenizer 对齐测试。
- 真实模型 smoke 尚未完成，当前为 synthetic GGUF metadata/tokenizer 验证。
- 不支持 Hunyuan-A13B/Large MoE 的专家路由、共享专家、top-k expert dispatch。
- 不支持 Hunyuan-VL/OCR/Video/Image 的视觉 projector 和多模态 token。
