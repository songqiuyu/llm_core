# CUDA 后端实现报告

## Summary

本轮新增实验 CUDA 后端，目标服务器为 RTX 4080 / `sm_89`。默认 CPU 构建不依赖 CUDA；服务器上通过 `-DAXONFORGE_ENABLE_CUDA=ON` 启用 CUDA Runtime + cuBLAS。

当前版本完成了 GPU 后端底座和第一批 kernel 原型，为后续 Qwen2.5-3B 的 GPU 常驻 forward 做准备。

## 已实现

- CMake：新增 `AXONFORGE_ENABLE_CUDA` 和 `AXONFORGE_CUDA_ARCH`，CUDA build 链接 `CUDA::cudart` 与 `CUDA::cublas`。
- Backend：新增 `cuda` backend，支持 device 初始化、stream、cuBLAS handle、显存分配、CPU/GPU 拷贝和同步。
- Tensor：新增 `Tensor::from_storage()`，允许 backend-owned storage 包装为 Tensor。
- Kernels：新增 F32 RMSNorm、SILU、MUL、RoPE，以及 correctness-first 的 Q4_K_M×F32 fused dequant GEMV 原型。
- Tests：CUDA build 下新增 backend 注册/拷贝测试和 RMSNorm kernel reference 对齐测试。
- CLI：`-b cuda` 可创建 CUDA backend；LLaMA-like 生成入口已预留 CUDA 分发点。

## 当前限制

- 完整 GPU-resident decoder forward 尚未完成；当前 `-b cuda -V` 会提示 CUDA 路径实验状态，并回退 CPU forward 保证文本生成正确。
- Q4_K_M GEMV kernel 是正确性优先版本，尚未进行 Ada Tensor Core / shared-memory tiling / CUDA Graph 优化。
- 尚未实现 GPU KV cache、GPU 权重缓存、batched prefill GEMM 和 CUDA sampling。

## GPU 服务器构建文档

以下步骤面向 RTX 4080 / CUDA Toolkit 服务器。当前推荐单独使用 `build-cuda/`，不要覆盖已有 CPU `build/` 目录。

### 1. 环境检查

```bash
nvidia-smi
nvcc --version
cmake --version
g++ --version
```

期望：

- `nvidia-smi` 能看到 RTX 4080，Driver 正常。
- `nvcc --version` 可用；如果没有 `nvcc`，需要安装 CUDA Toolkit，不只是 NVIDIA Driver。
- CMake 建议 3.20+，GCC 建议 11+。

如果服务器没有 `nvcc`，Ubuntu 常见安装方式如下：

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build
# CUDA Toolkit 建议按 NVIDIA 官方仓库安装；如果系统源可用，也可以先用：
sudo apt install -y nvidia-cuda-toolkit
```

安装后重新打开 shell，确认：

```bash
which nvcc
ldconfig -p | grep -E 'libcudart|libcublas'
```

### 2. 配置 CUDA Release 构建

RTX 4080 是 Ada 架构，CUDA arch 使用 `89`：

```bash
cmake -S . -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DAXONFORGE_ENABLE_CUDA=ON \
    -DAXONFORGE_CUDA_ARCH=89

cmake --build build-cuda -j$(nproc)
```

如果你偏好 Ninja：

```bash
cmake -S . -B build-cuda -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DAXONFORGE_ENABLE_CUDA=ON \
    -DAXONFORGE_CUDA_ARCH=89

cmake --build build-cuda
```

### 3. 验证 CUDA backend 已注册

```bash
build-cuda/tools/cli/axonforge-cli --list-backends
```

期望输出包含：

```text
cuda
cpu_x86
```

如果没有 `cuda`，说明 CMake 没有用 `-DAXONFORGE_ENABLE_CUDA=ON` 配置，或 CUDA Toolkit 未被 CMake 找到。

### 4. 运行 CUDA 单元测试

```bash
ctest --test-dir build-cuda --output-on-failure
```

CUDA build 会额外编译 `test_backend_cuda`，覆盖：

- CUDA backend 注册与初始化。
- CPU↔GPU tensor copy。
- CUDA RMSNorm kernel 与 CPU reference 对齐。

### 5. 下载首验模型

```bash
mkdir -p models/qwen2.5-3B

hf download Qwen/Qwen2.5-3B-Instruct-GGUF \
    qwen2.5-3b-instruct-q4_k_m.gguf \
    --local-dir models/qwen2.5-3B
```

如果没有 `hf` 命令：

```bash
python3 -m pip install -U huggingface_hub
```

### 6. 运行 AxonForge CUDA smoke

```bash
build-cuda/tools/cli/axonforge-cli \
    -b cuda \
    -m models/qwen2.5-3B/qwen2.5-3b-instruct-q4_k_m.gguf \
    -p "The capital of France is" -n 64 -t 0.0 -V
```

当前 v1 期望行为：

- stderr 打印 CUDA backend 信息，例如 GPU 名称、`sm_89`、显存大小。
- `-V` 下提示 CUDA generation path 仍为 experimental。
- 生成结果仍应正确；完整 GPU-resident forward 完成前会回退 CPU forward。

### 7. 收集 benchmark 日志

建议把配置、测试、运行日志都保存下来，方便后续调 kernel：

```bash
mkdir -p logs/cuda

nvidia-smi > logs/cuda/nvidia-smi.txt
cmake --build build-cuda -j$(nproc) 2>&1 | tee logs/cuda/build.log
ctest --test-dir build-cuda --output-on-failure 2>&1 | tee logs/cuda/ctest.log

/usr/bin/time -v build-cuda/tools/cli/axonforge-cli \
    -b cuda \
    -m models/qwen2.5-3B/qwen2.5-3b-instruct-q4_k_m.gguf \
    -p "The capital of France is" -n 64 -t 0.0 -V \
    2>&1 | tee logs/cuda/axonforge-qwen25-3b-cuda.log
```

同时开一个终端记录显存变化：

```bash
nvidia-smi dmon -s pucvmt -d 1 -o DT \
    > logs/cuda/nvidia-smi-dmon.log
```

### 8. llama.cpp CUDA 公平 baseline

使用同一 GGUF、同一 prompt、同一 context cap、同一生成 token 数：

```bash
./llama.cpp/build/bin/llama-cli \
    -m models/qwen2.5-3B/qwen2.5-3b-instruct-q4_k_m.gguf \
    -p "The capital of France is" \
    -n 64 -t 0.0 \
    -c 4096 -ngl 999 \
    2>&1 | tee logs/cuda/llamacpp-qwen25-3b-cuda.log
```

记录字段：

- AxonForge：TTFT、prefill tok/s、decode tok/s、RSS、CUDA backend 初始化信息。
- llama.cpp：prompt eval time、eval time、tok/s、VRAM 使用。
- 两边必须固定同一模型文件和同一 prompt。

### 9. 常见问题

`No CMAKE_CUDA_COMPILER could be found`：只安装了 Driver，没有 CUDA Toolkit；安装 `nvcc` 后重新配置。

`Could not find CUDAToolkit`：检查 `CUDA_HOME` 和 `PATH`：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
```

`BackendRegistry: no backend registered with id 'cuda'`：当前可执行文件不是 `build-cuda/` 下的，或配置时没开 `AXONFORGE_ENABLE_CUDA=ON`。

`invalid device function`：`AXONFORGE_CUDA_ARCH` 与 GPU 不匹配；RTX 4080 使用 `89`，A100 使用 `80`，3090 使用 `86`。

`out of memory`：先保持 `-c 4096` 或默认 context cap，不要直接使用模型 metadata 中的超长 context。

## 下一步

1. 将 Qwen2.5-3B 的权重上传并缓存到 GPU，按层建立 CUDA weight view。
2. 实现 GPU KV cache 和 single-token decode forward，先覆盖 Qwen2/Qwen2.5。
3. 用 CUDA Graph 捕获 decode token step，降低 kernel launch overhead。
4. 引入 CUTLASS 可选路径，对 FP16 fallback 和 dequant GEMM 做 Tensor Core 化。
