#include "axonforge/backend.hpp"
#include <immintrin.h>
#include <cmath>
#include <memory>

namespace axonforge::cpu_x86 {

// ============================================================
// Element-wise kernels (F32) — AVX2 path
// Processes 8 floats per cycle via 256-bit YMM registers.
// ============================================================

// ---- SoftMax (single-row, in-place) ----
class SoftmaxAvx2Kernel final : public IKernel {
public:
    void execute(std::span<const Tensor> inputs,
                 std::span<Tensor>       outputs,
                 const AttributeMap&     attrs) override {
        (void)attrs;
        // inputs[0] = logits [N]   outputs[0] = probs [N]
        const float* src = inputs[0].data_as<float>().data();
        float*       dst = outputs[0].data_as<float>().data();
        int64_t      N   = inputs[0].numel();

        // 1. Find max (for numerical stability)
        float max_val = src[0];
        for (int64_t i = 1; i < N; ++i) max_val = std::max(max_val, src[i]);

        // 2. exp(x - max) + sum
        // TODO(Phase0): vectorise with AVX2 (use polynomial exp approximation)
        float sum = 0.0f;
        for (int64_t i = 0; i < N; ++i) {
            dst[i] = std::exp(src[i] - max_val);
            sum += dst[i];
        }

        // 3. Normalise
        float inv_sum = 1.0f / sum;
        for (int64_t i = 0; i < N; ++i) dst[i] *= inv_sum;
        // TODO(Phase0): replace loops 1-3 with AVX2 vectorised version
    }
};

// ---- RMSNorm ----
class RmsNormAvx2Kernel final : public IKernel {
public:
    void execute(std::span<const Tensor> inputs,
                 std::span<Tensor>       outputs,
                 const AttributeMap&     attrs) override {
        // inputs[0] = x [N]   inputs[1] = weight [N]
        // outputs[0] = y [N]
        const float* x = inputs[0].data_as<float>().data();
        const float* w = inputs[1].data_as<float>().data();
        float*       y = outputs[0].data_as<float>().data();
        int64_t      N = inputs[0].numel();
        float eps = attrs.get_or<float>("eps", 1e-5f);

        // sum of squares
        float ss = 0.0f;
        for (int64_t i = 0; i < N; ++i) ss += x[i] * x[i];
        float rms = std::sqrt(ss / static_cast<float>(N) + eps);
        float inv_rms = 1.0f / rms;

        for (int64_t i = 0; i < N; ++i) y[i] = x[i] * inv_rms * w[i];
        // TODO(Phase0): AVX2 vectorised version using _mm256_mul_ps + horizontal add
    }
};

// ---- Generic element-wise (ADD, MUL, RELU, SILU, GELU) ----
class ElementwiseAvx2Kernel final : public IKernel {
public:
    explicit ElementwiseAvx2Kernel(OpType op) : op_(op) {}

    void execute(std::span<const Tensor> inputs,
                 std::span<Tensor>       outputs,
                 const AttributeMap&     /*attrs*/) override {
        float*       y = outputs[0].data_as<float>().data();
        int64_t      N = outputs[0].numel();
        const float* a = inputs[0].data_as<float>().data();
        const float* b = (inputs.size() > 1) ? inputs[1].data_as<float>().data() : nullptr;

        switch (op_) {
            case OpType::ADD:
                for (int64_t i = 0; i < N; ++i) y[i] = a[i] + b[i];
                break;
            case OpType::MUL:
                for (int64_t i = 0; i < N; ++i) y[i] = a[i] * b[i];
                break;
            case OpType::RELU:
                for (int64_t i = 0; i < N; ++i) y[i] = a[i] > 0.f ? a[i] : 0.f;
                break;
            case OpType::SILU:
                for (int64_t i = 0; i < N; ++i)
                    y[i] = a[i] / (1.0f + std::exp(-a[i]));  // x * sigmoid(x)
                break;
            case OpType::GELU:
                for (int64_t i = 0; i < N; ++i)
                    y[i] = 0.5f * a[i] * (1.0f + std::tanh(
                        0.7978845608f * (a[i] + 0.044715f * a[i] * a[i] * a[i])));
                break;
            default:
                throw std::runtime_error("ElementwiseAvx2: unsupported op");
        }
        // TODO(Phase0): replace all with AVX2 _mm256_* intrinsics
    }

private:
    OpType op_;
};

// ---- Factory functions (called from x86_backend.cpp) ----

std::unique_ptr<IKernel> make_softmax_avx2(const AttributeMap&) {
    return std::make_unique<SoftmaxAvx2Kernel>();
}
std::unique_ptr<IKernel> make_rms_norm_avx2(const AttributeMap&) {
    return std::make_unique<RmsNormAvx2Kernel>();
}
std::unique_ptr<IKernel> make_elementwise_avx2(OpType op, const AttributeMap&) {
    return std::make_unique<ElementwiseAvx2Kernel>(op);
}

} // namespace axonforge::cpu_x86
