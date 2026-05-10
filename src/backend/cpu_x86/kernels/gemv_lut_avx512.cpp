#include "axonforge/backend.hpp"
#include <memory>

#ifdef AXONFORGE_HAS_AVX512
#include <immintrin.h>

namespace axonforge::cpu_x86 {

// ============================================================
// T-MAC LUT GEMV — Q4_K_M × F32 → F32  (AVX-512 VNNI path)
//
// Processes 16 weights per VNNI instruction (_mm512_dpbusd_epi32),
// giving ~2x throughput vs AVX2 path on Intel Ice Lake+.
//
// TODO(Phase1): implement full kernel.
// ============================================================
class GemvQ4KmAvx512Kernel final : public IKernel {
public:
    void execute(std::span<const Tensor> /*inputs*/,
                 std::span<Tensor>       /*outputs*/,
                 const AttributeMap&     /*attrs*/) override {
        throw std::runtime_error("GemvQ4KmAvx512: not yet implemented");
    }
};

std::unique_ptr<IKernel> make_gemv_q4km_avx512(const AttributeMap&) {
    return std::make_unique<GemvQ4KmAvx512Kernel>();
}

} // namespace axonforge::cpu_x86

#endif // AXONFORGE_HAS_AVX512
