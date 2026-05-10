#include "axonforge/backend.hpp"
#include <memory>

#ifdef AXONFORGE_HAS_AVX512
#include <immintrin.h>

namespace axonforge::cpu_x86 {

// ============================================================
// INT8 GEMM via AVX-512 VNNI (_mm512_dpbusd_epi32)
// Used for W8A8 quantisation path.
//
// TODO(Phase1): implement.
// ============================================================

} // namespace axonforge::cpu_x86

#endif // AXONFORGE_HAS_AVX512
