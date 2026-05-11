#pragma once
#include <cstdint>
#include <cstddef>

// ─── Q8_0 block ───────────────────────────────────────────────────────────────
// 32 int8 quantised values with a single F32 scale.
// d = absmax(x[0..31]) / 127;  x[i] ≈ d * qs[i]
struct block_q8_0 {
    float  d;        // scale
    int8_t qs[32];   // quantised values in [-127, 127]
};
static_assert(sizeof(block_q8_0) == 36, "block_q8_0 layout changed");

// Number of Q8_0 blocks needed to cover k floats (k must be a multiple of 32)
inline constexpr std::size_t q8_blocks(int k) { return (std::size_t)(k / 32); }
