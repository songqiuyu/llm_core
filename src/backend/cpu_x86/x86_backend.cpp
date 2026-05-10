#include "axonforge/backend.hpp"
#include "cpuid.hpp"
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace axonforge::cpu_x86 {

// ============================================================
// Forward-declare kernel factory functions (defined in kernels/).
// Each returns nullptr if the required ISA is unavailable at runtime.
// ============================================================
std::unique_ptr<IKernel> make_gemm_f32_avx2(const AttributeMap&);
std::unique_ptr<IKernel> make_gemv_q4km_avx2(const AttributeMap&);
std::unique_ptr<IKernel> make_softmax_avx2(const AttributeMap&);
std::unique_ptr<IKernel> make_rms_norm_avx2(const AttributeMap&);
std::unique_ptr<IKernel> make_elementwise_avx2(OpType, const AttributeMap&);

// ---- AVX-512 kernels (compiled separately with -mavx512f -mavx512vnni) ----
#ifdef AXONFORGE_HAS_AVX512
std::unique_ptr<IKernel> make_gemv_q4km_avx512(const AttributeMap&);
std::unique_ptr<IKernel> make_gemm_int8_vnni(const AttributeMap&);
#endif

// ============================================================
// X86Backend — IBackend implementation for x86 Linux/Windows.
//
// Kernel selection strategy (highest ISA first):
//   AVX-512 VNNI → AVX2 → SSE4.2 (fallback, slow)
// ============================================================
class X86Backend final : public IBackend {
public:
    X86Backend()  = default;
    ~X86Backend() override = default;

    std::string_view name()      const noexcept override { return "cpu_x86"; }
    DeviceId         device_id() const noexcept override { return DeviceId::cpu(); }

    bool initialize() override {
        const auto& f = cpu_features();
        if (f.avx2) {
            printf("[AxonForge] x86 backend: AVX2%s%s enabled\n",
                   f.avx512f    ? " + AVX-512F"   : "",
                   f.avx512vnni ? " + VNNI"        : "");
        } else {
            printf("[AxonForge] x86 backend: SSE4.2 fallback (AVX2 not available)\n");
        }
        return true;
    }

    void finalize() override {}

    std::shared_ptr<TensorStorage>
    allocate(size_t bytes, size_t alignment) override {
        return std::make_shared<CpuTensorStorage>(bytes, alignment);
    }

    void transfer(const Tensor& src, Tensor& dst) override {
        if (!src.is_contiguous() || !dst.is_contiguous()) {
            throw std::runtime_error("X86Backend::transfer: both tensors must be contiguous");
        }
        if (src.nbytes() != dst.nbytes()) {
            throw std::runtime_error("X86Backend::transfer: size mismatch");
        }
        std::memcpy(dst.raw_data(), src.raw_data(), src.nbytes());
    }

    void synchronize() override {} // CPU is always synchronous

    bool can_handle(OpType op, std::span<const DType> input_dtypes) const override {
        // x86 backend handles all ops on CPU tensors
        (void)input_dtypes;
        switch (op) {
            case OpType::MATMUL:
            case OpType::ADD: case OpType::SUB: case OpType::MUL: case OpType::DIV:
            case OpType::RELU: case OpType::SILU: case OpType::GELU:
            case OpType::SOFTMAX:
            case OpType::RMS_NORM: case OpType::LAYER_NORM:
            case OpType::ROPE:
            case OpType::EMBEDDING_LOOKUP:
            case OpType::RESHAPE: case OpType::PERMUTE: case OpType::SLICE:
            case OpType::CAST:
                return true;
            default:
                return false;
        }
    }

    std::unique_ptr<IKernel>
    get_kernel(OpType op,
               std::span<const DType> input_dtypes,
               const AttributeMap& attrs) override {
        const auto& f = cpu_features();
        (void)input_dtypes;

        switch (op) {
            case OpType::MATMUL: {
#ifdef AXONFORGE_HAS_AVX512
                if (f.avx512vnni && !input_dtypes.empty() &&
                    input_dtypes[0] == DType::Q4_K_M)
                    return make_gemv_q4km_avx512(attrs);
#endif
                if (f.avx2) {
                    if (!input_dtypes.empty() && input_dtypes[0] == DType::Q4_K_M)
                        return make_gemv_q4km_avx2(attrs);
                    return make_gemm_f32_avx2(attrs);
                }
                throw std::runtime_error("X86Backend: MATMUL requires at least AVX2");
            }
            case OpType::SOFTMAX:
                if (f.avx2) return make_softmax_avx2(attrs);
                throw std::runtime_error("X86Backend: SOFTMAX requires AVX2");
            case OpType::RMS_NORM:
                if (f.avx2) return make_rms_norm_avx2(attrs);
                throw std::runtime_error("X86Backend: RMS_NORM requires AVX2");
            case OpType::ADD: case OpType::MUL: case OpType::RELU:
            case OpType::SILU: case OpType::GELU:
                if (f.avx2) return make_elementwise_avx2(op, attrs);
                throw std::runtime_error("X86Backend: elementwise op requires AVX2");
            default:
                throw std::runtime_error(
                    std::string("X86Backend: unsupported op: ") +
                    std::string(op_type_name(op)));
        }
    }
};

} // namespace axonforge::cpu_x86

// ---- Self-registration (runs before main()) ----
// [[gnu::constructor]] ensures this runs before any static initializer
// that might call BackendRegistry::instance().create().
[[gnu::constructor]] static void axonforge_register_x86_backends() {
    auto& reg = ::axonforge::BackendRegistry::instance();
    reg.register_backend("cpu_x86",
        []() -> std::unique_ptr<::axonforge::IBackend> {
            return std::make_unique<axonforge::cpu_x86::X86Backend>();
        });
    // "cpu" is a convenience alias that always maps to the best available CPU backend
    reg.register_backend("cpu",
        []() -> std::unique_ptr<::axonforge::IBackend> {
            return std::make_unique<axonforge::cpu_x86::X86Backend>();
        });
}
