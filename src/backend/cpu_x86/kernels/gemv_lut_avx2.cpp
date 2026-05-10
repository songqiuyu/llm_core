#include "axonforge/backend.hpp"
#include <immintrin.h>
#include <memory>

namespace axonforge::cpu_x86 {

// ============================================================
// T-MAC-style LUT GEMV — Q4_K_M × F32 → F32  (AVX2 path)
//
// Key idea (from T-MAC paper):
//   For a 4-bit weight w[i] ∈ {0..15}, and activation x[j]:
//     dot product = Σ_b (2^b * sign_table[b][i]) * (tbl[w[i]] * x[j])
//
//   Replaces multiply-accumulate with table-lookup + shift + add,
//   eliminating all FP multiplications for quantized weights.
//
// AVX2 implementation plan:
//   1. Pre-build a 16-entry LUT of (dequant_scale * x) per 32-elem block
//   2. Use _mm256_i32gather to batch-lookup 8 weights at once
//   3. Accumulate with _mm256_fmadd_ps
//
// TODO(Phase1): implement full T-MAC LUT GEMV kernel.
// ============================================================
class GemvQ4KmAvx2Kernel final : public IKernel {
public:
    void execute(std::span<const Tensor> inputs,
                 std::span<Tensor>       outputs,
                 const AttributeMap&     /*attrs*/) override {
        // inputs[0] = x [N]  (F32 activation vector)
        // inputs[1] = W [M, N/2] (Q4_K_M packed weights)
        // outputs[0] = y [M] (F32 output)
        (void)inputs; (void)outputs;
        // TODO(Phase1): implement T-MAC LUT GEMV
        throw std::runtime_error("GemvQ4KmAvx2: not yet implemented");
    }
};

std::unique_ptr<IKernel> make_gemv_q4km_avx2(const AttributeMap&) {
    return std::make_unique<GemvQ4KmAvx2Kernel>();
}

} // namespace axonforge::cpu_x86
