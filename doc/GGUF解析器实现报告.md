# GGUF 解析器实现报告

> **状态**：完成，47/47 单元测试通过  
> **日期**：2026-05-10  
> **对应 Phase**：Phase 0（GGUF 加载 + Engine::from_gguf）

---

## 1. 背景与目标

GGUF（Generic GPU Format Unified）是 llama.cpp / ggml 生态的标准模型文件格式。  
一个 `.gguf` 文件将**模型超参数**（架构、层数、注意力头数等）、**分词器词表**和**全部权重张量**打包进单个文件，支持 `mmap` 零拷贝加载。

本阶段目标：

1. 实现 GGUF v2/v3 二进制解析器 `GgufReader`，支持完整的 KV 元数据和 TensorInfo 解析。
2. 将解析结果接入 `Engine::from_gguf()`，填充 `ModelConfig`（架构超参数）并构建零拷贝权重 `Tensor` 视图。
3. 用合成 GGUF 二进制驱动的白盒单元测试覆盖正常路径与所有错误路径。

---

## 2. GGUF 文件格式回顾

```
[4 B]  magic = 'G','G','U','F'
[4 B]  version (uint32_t, v2 或 v3)
[8 B]  n_tensors (int64_t)
[8 B]  n_kv      (int64_t)

for i in 0..n_kv:
    [len+8 B]  key   (string: uint64 len + chars)
    [4 B]      type  (int32_t, gguf_type)
    [...]      value
               如果 type == ARRAY(9):
                   [4 B] elem_type (int32_t)
                   [8 B] count (uint64_t)
                   [...]  count × elem

for i in 0..n_tensors:
    [len+8 B]  name      (string)
    [4 B]      n_dims    (uint32_t)
    n_dims × [8 B]       shape dim (int64_t each)
    [4 B]      ggml_type (int32_t)
    [8 B]      offset    (uint64_t, within data blob)

[pad]  对齐到 general.alignment（默认 32 字节）
[...]  tensor data blob
```

关键细节（从 ggml 源码确认）：

| 字段 | 文件类型 |
|------|---------|
| version | uint32_t |
| n_tensors / n_kv | **int64_t**（非 uint64_t） |
| gguf_type / ggml_type | **int32_t**（非枚举宽度） |
| bool 值 | int8_t（非标准 bool） |
| string 长度 | uint64_t |
| 数组 count | uint64_t |
| tensor 维度 | int64_t |

---

## 3. 实现方案

### 3.1 解析策略：单遍游标 + mmap

```
open(path)
  → fstat → mmap(PROT_READ, MAP_PRIVATE)
     → 设置 cursor_ = begin_ = (uint8_t*)mmap_ptr_
        → parse()
           → 校验 magic[4] == "GGUF"
           → read version(u32), n_tensors(i64), n_kv(i64)
           → for n_kv:  read_string() → read_kv_value(type_tag)
           → for n_tensors: read name/shape/ggml_type/offset → GgufTensorInfo
           → tensor_data_base_ = ALIGN(cursor_ - begin_, alignment_)
```

所有读取均通过 `check_bounds(n)` 在推进游标前验证剩余空间，截断文件立即抛出 `std::runtime_error`。

### 3.2 GgufValue 类型

使用 `std::variant` 表示 13 种 GGUF KV 值类型：

```cpp
using GgufValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, float, bool, std::string,
    GgufArray,   // type=9 (ARRAY)
    uint64_t, int64_t, double
>;
```

`GgufArray` 分两条路径：
- `elem_type == STRING(8)`：读 N 个变长字符串 → `strings` 字段
- 其他 POD 类型：计算 `count × esz` 字节，拷贝到 `data` 字段

### 3.3 KV 访问器：宽化转换

```cpp
uint64_t get_u64(key, def)   // 接受 u8/i8/u16/i16/u32/i32/i64/u64
uint32_t get_u32(key, def)   // 转为 u64 后截断
int32_t  get_i32(key, def)
float    get_f32(key, def)
std::string get_str(key, def)
```

真实 GGUF 文件中同一字段可能写 `uint32_t` 或 `uint64_t`（取决于 quantize 工具版本），宽化转换保证兼容性。

### 3.4 零拷贝权重 Tensor

```cpp
Tensor GgufReader::weight_tensor(size_t idx) const {
    const auto& info = tensors_[idx];
    void* ptr = begin_ + tensor_data_base_ + info.offset;   // 直接指向 mmap
    return Tensor::from_raw_blob(ptr, info.nbytes, info.shape, info.dtype);
}
```

- `from_raw_blob` 是本次新增的 `Tensor` 工厂，用显式 `nbytes` 包装 `CpuTensorStorage(owns=false)`。
- **零字节复制**：权重数据不移动，OS 按 page fault 懒加载。
- **生命周期**：`Engine::Impl` 持有 `unique_ptr<GgufReader>`，`GgufReader` 析构时 `munmap`；所有权重 Tensor 的 storage 因 `owns=false` 不会尝试 free。

### 3.5 DType 扩展（dtype.hpp）

新增三个内联函数：

| 函数 | 说明 |
|------|------|
| `ggml_type_to_dtype(int32_t)` | ggml_type → DType 映射（11 个已知类型） |
| `gguf_type_pod_size(int32_t)` | KV 标量字节大小（0 表示变长） |
| `gguf_tensor_nbytes(DType, int64_t)` | 精确权重字节数（含 block-quantized 块边界） |

量化类型块大小（来自 GGML 标准）：

| 类型 | block_size | bytes/block |
|------|-----------|-------------|
| Q8_0 | 32 | 34 |
| Q4_0 | 32 | 18 |
| Q4_K | 256 | 144 |
| Q3_K | 256 | 110 |
| Q2_K | 256 | 84 |

### 3.6 Engine::from_gguf 组装

```
GgufReader::open()
  → extract ModelConfig from KV (arch, n_layers, hidden_dim, ...)
  → bos/eos token ids from tokenizer.ggml.*_token_id
  → BackendRegistry::instance().create(cfg.backend)
  → build weights map (name → Tensor view)
  → Engine::Impl { engine_cfg, model_cfg, backend, gguf, weights, bos/eos }
  → return Engine(impl)
```

---

## 4. 文件变更一览

| 文件 | 变更 |
|------|------|
| `include/axonforge/dtype.hpp` | 新增 `ggml_type_to_dtype`, `gguf_type_pod_size`, `gguf_tensor_nbytes` |
| `include/axonforge/tensor.hpp` | 新增 `Tensor::from_raw_blob()` 声明 |
| `src/core/tensor.cpp` | 新增 `Tensor::from_raw_blob()` 实现 |
| `src/loader/gguf_reader.hpp` | **新建** — `GgufValue`, `GgufArray`, `GgufTensorInfo`, `GgufReader` 声明 |
| `src/loader/gguf_loader.cpp` | **全部重写** — `GgufReader` 完整实现（~350 行） |
| `src/engine.cpp` | `Engine::Impl` 扩展字段；新增 `Engine::from_gguf()` 实现；修正 `bos_id/eos_id` |
| `tests/test_gguf_reader.cpp` | **新建** — 13 个测试用例 |
| `tests/CMakeLists.txt` | 注册 `test_gguf_reader` |

---

## 5. 测试结果

```
47 tests passed, 0 failed  （2026-05-10）

test_tensor      [17 cases]  不变
test_graph       [ 8 cases]  不变
test_backend_x86 [ 9 cases]  不变
test_gguf_reader [13 cases]  ← 新增
  ├─ ggml_type_to_dtype maps known types
  ├─ gguf_tensor_nbytes exact sizes
  ├─ gguf_type_pod_size covers all 13 types
  ├─ Engine::from_gguf parses architecture KV
  ├─ Engine::from_gguf reads LLaMA hyperparams
  ├─ Engine::from_gguf sets bos/eos token ids
  ├─ Engine::from_gguf reads vocab_size from tokens array
  ├─ Engine::from_gguf loads one tensor with correct shape and dtype
  ├─ Engine::from_gguf tensor data is accessible (zero-copy check)
  ├─ Engine::from_gguf throws on non-existent file
  ├─ Engine::from_gguf throws on invalid magic
  ├─ Engine::from_gguf throws on truncated file
  └─ Engine::from_gguf throws on bad GGUF version
```

---

## 6. 已知限制 / 下一步

| 项目 | 状态 | 计划 |
|------|------|------|
| 大端序 GGUF | 不支持（规范中极罕见） | Phase 2 可加 bswap |
| 词表 BPE 解码 | 仅读 tokens 数组计 vocab_size | Phase 0 续集：bpe_tokenizer.cpp |
| LLaMA 图构建 | 权重已就绪，图尚未接线 | Phase 0 续集：llama_model.cpp |
| 量化 dequant | 权重存 raw bytes，dequant 在 kernel 层 | Phase 1 |
| 多文件 shard | 不支持 | Phase 2 |
| ARRAY of ARRAY | 抛出异常（协议中合法但实际未出现） | 按需支持 |
