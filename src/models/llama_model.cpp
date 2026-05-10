#include "axonforge/graph.hpp"
#include "axonforge/engine.hpp"

namespace axonforge {
// ============================================================
// LLaMA-3 / LLaMA-2 Model Graph Definition
//
// Architecture (one transformer layer):
//   x  →  RMSNorm  →  QKV_proj  →  RoPE  →  Attention  →  o_proj
//      →  residual_add
//      →  RMSNorm  →  gate_proj,up_proj  →  SiLU(gate)*up  →  down_proj
//      →  residual_add
//
// GraphBuilder call sequence (per-layer):
//   residual = x
//   x = rms_norm(x, attn_norm_w)
//   q = matmul(x, Wq)     [seq, n_heads * head_dim]
//   k = matmul(x, Wk)     [seq, n_kv_heads * head_dim]
//   v = matmul(x, Wv)     [seq, n_kv_heads * head_dim]
//   q = rope(q, freqs, head_dim)
//   k = rope(k, freqs, head_dim)
//   attn_out = scaled_dot_product_attn(q, k, v, kv_cache)
//   x = matmul(attn_out, Wo) + residual
//   residual = x
//   x = rms_norm(x, ffn_norm_w)
//   gate = silu(matmul(x, W_gate))
//   up   = matmul(x, W_up)
//   x = matmul(gate * up, W_down) + residual
//
// TODO(Phase0): build the full LLaMA-3 graph with GraphBuilder.
// ============================================================

void build_llama_graph(ComputeGraph& /*graph*/, const ModelConfig& /*cfg*/) {
    // TODO(Phase0): implement
}

} // namespace axonforge
