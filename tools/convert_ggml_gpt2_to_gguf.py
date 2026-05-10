#!/usr/bin/env python3
"""
Convert GPT-2 GGML binary (ggml-model.bin) to GGUF v3.

Old GGML format layout:
  [4B]  magic   = 0x67676d6c ('ggml' LE)
  [4B]  n_vocab (int32)
  [4B]  n_ctx   (int32)
  [4B]  n_embd  (int32)
  [4B]  n_head  (int32)
  [4B]  n_layer (int32)
  [4B]  ftype   (int32)
  [4B]  n_vocab (int32, repeated in vocab section)
  n_vocab × { [4B] len(uint32) + [len B] token_bytes }
  tensors:
    { [4B] n_dims(i32) [4B] namelen(i32) [4B] ttype(i32)
      n_dims×[4B] ne[i](i32)  [namelen B] name  [data B] }

Usage:
  python3 tools/convert_ggml_gpt2_to_gguf.py \\
      --input  models/gpt-2-117M/ggml-model.bin \\
      --output models/gpt-2-117M/gpt2-117M-f16.gguf
"""

import struct
import sys
import argparse

GGML_FILE_MAGIC         = 0x67676d6c
GGML_QNT_VERSION_FACTOR = 1000
GGUF_VERSION            = 3
GGUF_ALIGNMENT          = 32

# GGUF KV value-type tags
KV_UINT32  = 4
KV_FLOAT32 = 6
KV_STRING  = 8
KV_ARRAY   = 9

# GGML tensor type → byte-size per element (scalar types only)
GGML_NBYTES = {0: 4, 1: 2}  # F32=4, F16=2


# ─── GGUF serialisation helpers ──────────────────────────────────────────────

def gguf_str(s: str) -> bytes:
    enc = s.encode("utf-8")
    return struct.pack("<Q", len(enc)) + enc


def gguf_kv_str(key: str, val: str) -> bytes:
    return gguf_str(key) + struct.pack("<i", KV_STRING) + gguf_str(val)


def gguf_kv_u32(key: str, val: int) -> bytes:
    return gguf_str(key) + struct.pack("<iI", KV_UINT32, val)


def gguf_kv_f32(key: str, val: float) -> bytes:
    return gguf_str(key) + struct.pack("<if", KV_FLOAT32, val)


def gguf_kv_str_array(key: str, items: list) -> bytes:
    b = gguf_str(key)
    b += struct.pack("<i", KV_ARRAY)
    b += struct.pack("<i", KV_STRING)          # elem type = STRING
    b += struct.pack("<Q", len(items))          # count
    for item in items:
        b += gguf_str(item)
    return b


# ─── GGML reader ─────────────────────────────────────────────────────────────

def read_ggml(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read()

    pos = 0

    def u32():
        nonlocal pos
        v = struct.unpack_from("<I", raw, pos)[0]; pos += 4; return v

    def i32():
        nonlocal pos
        v = struct.unpack_from("<i", raw, pos)[0]; pos += 4; return v

    # magic
    magic = u32()
    if magic != GGML_FILE_MAGIC:
        sys.exit(f"ERROR: bad magic {magic:#010x}, expected {GGML_FILE_MAGIC:#010x}")

    # hyperparams
    n_vocab = i32()
    n_ctx   = i32()
    n_embd  = i32()
    n_head  = i32()
    n_layer = i32()
    ftype   = i32()
    ftype   = ftype % GGML_QNT_VERSION_FACTOR   # strip version prefix

    print(f"  n_vocab={n_vocab}  n_ctx={n_ctx}  n_embd={n_embd}"
          f"  n_head={n_head}  n_layer={n_layer}  ftype={ftype}")

    # vocab
    nv = i32()
    if nv != n_vocab:
        sys.exit(f"ERROR: vocab count mismatch {nv} vs {n_vocab}")
    tokens = []
    for _ in range(n_vocab):
        length = u32()
        word = raw[pos:pos + length].decode("utf-8", errors="replace")
        pos += length
        tokens.append(word)
    print(f"  Read {len(tokens)} vocabulary tokens")

    # tensors
    tensors = []          # list of (name, ne_list, ttype, bytes)
    while pos < len(raw):
        if pos + 12 > len(raw):
            break
        n_dims = i32()
        namelen = i32()
        ttype  = i32()
        ne = [i32() for _ in range(n_dims)]
        name = raw[pos:pos + namelen].decode("utf-8"); pos += namelen

        nelements = 1
        for d in ne: nelements *= d
        bpe = GGML_NBYTES.get(ttype, 0)
        if bpe == 0:
            sys.exit(f"ERROR: unsupported tensor type {ttype} for '{name}'")
        nbytes = nelements * bpe

        if pos + nbytes > len(raw):
            sys.exit(f"ERROR: file truncated while reading tensor '{name}'")
        tensor_data = raw[pos:pos + nbytes]; pos += nbytes

        tensors.append((name, ne, ttype, tensor_data))
        tname = "F16" if ttype == 1 else "F32"
        print(f"    {name:50s}  shape={ne}  dtype={tname}  {nbytes/1024:.1f} KB")

    return dict(n_vocab=n_vocab, n_ctx=n_ctx, n_embd=n_embd,
                n_head=n_head, n_layer=n_layer, ftype=ftype,
                tokens=tokens, tensors=tensors)


# ─── GGUF writer ─────────────────────────────────────────────────────────────

def write_gguf(model: dict, out_path: str):
    n_vocab  = model["n_vocab"]
    n_ctx    = model["n_ctx"]
    n_embd   = model["n_embd"]
    n_head   = model["n_head"]
    n_layer  = model["n_layer"]
    tokens   = model["tokens"]
    tensors  = model["tensors"]

    # ---- KV metadata ----
    kv  = gguf_kv_str("general.architecture",              "gpt2")
    kv += gguf_kv_str("general.name",                      "GPT-2 117M")
    kv += gguf_kv_u32("gpt2.context_length",               n_ctx)
    kv += gguf_kv_u32("gpt2.embedding_length",             n_embd)
    kv += gguf_kv_u32("gpt2.feed_forward_length",          4 * n_embd)
    kv += gguf_kv_u32("gpt2.attention.head_count",         n_head)
    kv += gguf_kv_u32("gpt2.block_count",                  n_layer)
    kv += gguf_kv_f32("gpt2.attention.layer_norm_epsilon", 1e-5)
    kv += gguf_kv_str("tokenizer.ggml.model",              "gpt2")
    kv += gguf_kv_str_array("tokenizer.ggml.tokens",       tokens)
    kv += gguf_kv_u32("tokenizer.ggml.bos_token_id",       50256)
    kv += gguf_kv_u32("tokenizer.ggml.eos_token_id",       50256)
    n_kv = 12

    # ---- TensorInfo + data offsets ----
    tensor_info = b""
    data_parts  = []
    data_off    = 0

    for name, ne, ttype, tdata in tensors:
        tensor_info += gguf_str(name)
        tensor_info += struct.pack("<I", len(ne))          # n_dims uint32
        for d in ne:
            tensor_info += struct.pack("<q", d)            # each dim int64
        tensor_info += struct.pack("<i", ttype)            # ggml_type int32
        tensor_info += struct.pack("<Q", data_off)         # offset uint64
        data_parts.append(tdata)
        data_off += len(tdata)

    n_tensors = len(tensors)

    # ---- Header ----
    header  = b"GGUF"
    header += struct.pack("<I", GGUF_VERSION)              # version uint32
    header += struct.pack("<q", n_tensors)                 # n_tensors int64
    header += struct.pack("<q", n_kv)                      # n_kv int64

    meta = header + kv + tensor_info

    # ---- Alignment padding ----
    pad = (GGUF_ALIGNMENT - len(meta) % GGUF_ALIGNMENT) % GGUF_ALIGNMENT
    meta += b"\x00" * pad

    # ---- Write ----
    total = len(meta) + data_off
    with open(out_path, "wb") as f:
        f.write(meta)
        for tdata in data_parts:
            f.write(tdata)

    print(f"\n  Wrote {total/1024/1024:.1f} MB → {out_path}")
    print(f"  metadata: {len(meta)/1024:.1f} KB  |  tensor data: {data_off/1024/1024:.1f} MB")
    print(f"  {n_tensors} tensors  |  {n_kv} KV entries")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Convert GPT-2 GGML → GGUF v3")
    ap.add_argument("--input",  required=True, help="Path to ggml-model.bin")
    ap.add_argument("--output", required=True, help="Output .gguf path")
    args = ap.parse_args()

    print(f"Reading GGML: {args.input}")
    model = read_ggml(args.input)

    print(f"\nWriting GGUF: {args.output}")
    write_gguf(model, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
