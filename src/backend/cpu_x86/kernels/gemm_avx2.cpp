#include "axonforge/backend.hpp"
#include <immintrin.h>
#include <cstring>
#include <memory>

namespace axonforge::cpu_x86 {

// ============================================================
// F32 GEMM — AVX2 + FMA
//
// Tiling strategy (to be tuned):
//   MC=64, KC=256, NC=1024  (L2 cache friendly)
//   Micro-kernel: 8×1 (8 FP32 per AVX2 register × 1 accumulator row)
//
// TODO(Phase0): implement full blocked GEMM with:
//   - PACK_A: row-major A → column-major panels
//   - PACK_B: column-major B → row-major panels
//   - Micro-kernel loop with FMA
// ============================================================
class GemmF32Avx2Kernel final : public IKernel {
public:
    void execute(std::span<const Tensor> inputs,
                 std::span<Tensor>       outputs,
                 const AttributeMap&     /*attrs*/) override {
        // inputs[0] = A [M, K]   inputs[1] = B [K, N]
        // outputs[0] = C [M, N]
        const Tensor& A = inputs[0];
        const Tensor& B = inputs[1];
        Tensor&       C = outputs[0];

        const int64_t M = A.dim(0);
        const int64_t K = A.dim(1);
        const int64_t N = B.dim(1);

        const float* a = A.data_as<float>().data();
        const float* b = B.data_as<float>().data();
        float*       c = C.data_as<float>().data();

        // Reference scalar fallback (replaced by AVX2 micro-kernel in Phase0)
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    acc += a[m * K + k] * b[k * N + n];
                }
                c[m * N + n] = acc;
            }
        }
        // TODO(Phase0): replace with AVX2 tiled implementation
    }
};

std::unique_ptr<IKernel> make_gemm_f32_avx2(const AttributeMap&) {
    return std::make_unique<GemmF32Avx2Kernel>();
}

} // namespace axonforge::cpu_x86
