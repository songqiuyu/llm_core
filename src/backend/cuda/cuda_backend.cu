#include "axonforge/backend.hpp"
#include "cuda_kernels.hpp"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace axonforge::cuda {
namespace {

static void check_cuda(cudaError_t st, const char* where) {
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(st));
    }
}

static void check_cublas(cublasStatus_t st, const char* where) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(where) + ": cuBLAS error " + std::to_string((int)st));
    }
}

class CudaTensorStorage final : public TensorStorage {
public:
    explicit CudaTensorStorage(size_t bytes) : bytes_(bytes) {
        if (bytes_ > 0) check_cuda(cudaMalloc(&ptr_, bytes_), "cudaMalloc");
    }
    ~CudaTensorStorage() override {
        if (ptr_) cudaFree(ptr_);
    }
    CudaTensorStorage(const CudaTensorStorage&) = delete;
    CudaTensorStorage& operator=(const CudaTensorStorage&) = delete;

    void* data() noexcept override { return ptr_; }
    const void* data() const noexcept override { return ptr_; }
    size_t byte_size() const noexcept override { return bytes_; }
    DeviceId device() const noexcept override { return DeviceId::cuda(0); }

private:
    void* ptr_{nullptr};
    size_t bytes_{0};
};

class CudaKernel final : public IKernel {
public:
    CudaKernel(OpType op, cudaStream_t stream) : op_(op), stream_(stream) {}

    void execute(std::span<const Tensor> inputs,
                 std::span<Tensor> outputs,
                 const AttributeMap& attrs) override {
        if (outputs.empty()) throw std::runtime_error("CudaKernel::execute: no output tensor");
        switch (op_) {
            case OpType::RMS_NORM: {
                if (inputs.size() < 2) throw std::runtime_error("CUDA RMS_NORM requires x and weight");
                const int n = static_cast<int>(inputs[0].numel());
                const float eps = attrs.get_or<float>("eps", 1e-5f);
                rms_norm_f32(static_cast<float*>(outputs[0].raw_data()),
                             static_cast<const float*>(inputs[0].raw_data()),
                             static_cast<const float*>(inputs[1].raw_data()),
                             n, eps, stream_);
                break;
            }
            case OpType::SILU: {
                if (inputs.empty()) throw std::runtime_error("CUDA SILU requires input");
                const int n = static_cast<int>(inputs[0].numel());
                silu_f32(static_cast<float*>(outputs[0].raw_data()),
                         static_cast<const float*>(inputs[0].raw_data()),
                         n, stream_);
                break;
            }
            case OpType::MUL: {
                if (inputs.size() < 2) throw std::runtime_error("CUDA MUL requires two inputs");
                const int n = static_cast<int>(outputs[0].numel());
                mul_f32(static_cast<float*>(outputs[0].raw_data()),
                        static_cast<const float*>(inputs[0].raw_data()),
                        static_cast<const float*>(inputs[1].raw_data()),
                        n, stream_);
                break;
            }
            case OpType::ROPE: {
                if (inputs.empty()) throw std::runtime_error("CUDA ROPE requires input/output alias tensor");
                const int n_heads = static_cast<int>(attrs.get_or<int64_t>("n_heads", 0));
                const int head_dim = static_cast<int>(attrs.get_or<int64_t>("head_dim", 0));
                const int pos = static_cast<int>(attrs.get_or<int64_t>("pos", 0));
                const float theta = attrs.get_or<float>("theta", 10000.0f);
                if (outputs[0].raw_data() != inputs[0].raw_data()) {
                    check_cuda(cudaMemcpyAsync(outputs[0].raw_data(), inputs[0].raw_data(),
                                               inputs[0].nbytes(), cudaMemcpyDeviceToDevice, stream_),
                               "cudaMemcpyAsync(ROPE)");
                }
                rope_f32(static_cast<float*>(outputs[0].raw_data()), n_heads, head_dim, pos, theta, stream_);
                break;
            }
            case OpType::MATMUL: {
                if (inputs.size() < 2) throw std::runtime_error("CUDA MATMUL requires weight and x");
                if (inputs[0].dtype() != DType::Q4_K_M || inputs[1].dtype() != DType::F32) {
                    throw std::runtime_error("CUDA MATMUL v1 supports Q4_K_M weight x F32 activation only");
                }
                const int rows = static_cast<int>(outputs[0].numel());
                const int cols = static_cast<int>(inputs[1].numel());
                const size_t row_bytes = attrs.get_or<int64_t>("row_bytes", (int64_t)((cols / 256) * 144));
                q4km_gemv_f32(static_cast<float*>(outputs[0].raw_data()),
                              static_cast<const uint8_t*>(inputs[0].raw_data()),
                              static_cast<const float*>(inputs[1].raw_data()),
                              rows, cols, row_bytes, stream_);
                break;
            }
            default:
                throw std::runtime_error("CudaKernel::execute: unsupported op");
        }
        check_cuda(cudaGetLastError(), "CUDA kernel launch");
    }

private:
    OpType op_;
    cudaStream_t stream_{};
};

class CudaBackend final : public IBackend {
public:
    CudaBackend() = default;
    ~CudaBackend() override { finalize(); }

    std::string_view name() const noexcept override { return "cuda"; }
    DeviceId device_id() const noexcept override { return DeviceId::cuda(0); }

    bool initialize() override {
        if (initialized_) return true;
        int count = 0;
        check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (count <= 0) return false;
        check_cuda(cudaSetDevice(0), "cudaSetDevice");
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        check_cublas(cublasCreate(&cublas_), "cublasCreate");
        check_cublas(cublasSetStream(cublas_, stream_), "cublasSetStream");
        check_cublas(cublasLtCreate(&cublas_lt_), "cublasLtCreate");
        cudaDeviceProp prop{};
        check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        std::printf("[AxonForge] CUDA backend: %s sm_%d%d, %.1f GiB VRAM\n",
                    prop.name, prop.major, prop.minor,
                    (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        initialized_ = true;
        return true;
    }

    void finalize() override {
        if (cublas_) {
            cublasDestroy(cublas_);
            cublas_ = nullptr;
        }
        if (cublas_lt_) {
            cublasLtDestroy(cublas_lt_);
            cublas_lt_ = nullptr;
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        initialized_ = false;
    }

    std::shared_ptr<TensorStorage> allocate(size_t bytes, size_t) override {
        return std::make_shared<CudaTensorStorage>(bytes);
    }

    void transfer(const Tensor& src, Tensor& dst) override {
        if (src.nbytes() != dst.nbytes()) {
            throw std::runtime_error("CudaBackend::transfer: size mismatch");
        }
        cudaMemcpyKind kind = cudaMemcpyDefault;
        check_cuda(cudaMemcpyAsync(dst.raw_data(), src.raw_data(), src.nbytes(), kind, stream_),
                   "cudaMemcpyAsync");
        synchronize();
    }

    void synchronize() override {
        if (stream_) check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

    bool can_handle(OpType op, std::span<const DType> input_dtypes) const override {
        switch (op) {
            case OpType::RMS_NORM:
            case OpType::SILU:
            case OpType::MUL:
            case OpType::ROPE:
                return input_dtypes.empty() || input_dtypes[0] == DType::F32;
            case OpType::MATMUL:
                return input_dtypes.size() >= 2 &&
                       input_dtypes[0] == DType::Q4_K_M &&
                       input_dtypes[1] == DType::F32;
            default:
                return false;
        }
    }

    std::unique_ptr<IKernel> get_kernel(OpType op,
                                        std::span<const DType> input_dtypes,
                                        const AttributeMap&) override {
        if (!can_handle(op, input_dtypes)) {
            throw std::runtime_error("CudaBackend::get_kernel: unsupported op/dtype combination");
        }
        return std::make_unique<CudaKernel>(op, stream_);
    }

private:
    bool initialized_{false};
    cudaStream_t stream_{};
    cublasHandle_t cublas_{};
    cublasLtHandle_t cublas_lt_{};
};

} // namespace
} // namespace axonforge::cuda

[[gnu::constructor]] static void axonforge_register_cuda_backend() {
    auto& reg = ::axonforge::BackendRegistry::instance();
    reg.register_backend("cuda",
        []() -> std::unique_ptr<::axonforge::IBackend> {
            return std::make_unique<::axonforge::cuda::CudaBackend>();
        });
}
