#!/usr/bin/env python3
"""Debug: compare our raw encoder vs llama.cpp tokenizer on the chat template."""
import sys
sys.path.insert(0, 'llama.cpp/gguf-py')
from gguf import GGUFReader

r = GGUFReader('models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
toks = None
for f in r.fields.values():
    if f.name == 'tokenizer.ggml.tokens':
        toks = [bytes(f.parts[p]).decode('utf-8', errors='replace') for p in f.data]
        break

rev = {t: i for i, t in enumerate(toks)}

byte_to_id = {}
for b in range(256):
    key = '<0x%02X>' % b
    if key in rev:
        byte_to_id[b] = rev[key]

SP = '\u2581'  # U+2581 LOWER ONE EIGHTH BLOCK (SentencePiece space marker)

BOS_ID, EOS_ID = 1, 2
BOS_STR = toks[BOS_ID]  # '<s>'
EOS_STR = toks[EOS_ID]  # '</s>'
SPECIALS = sorted([(BOS_STR, BOS_ID), (EOS_STR, EOS_ID)], key=lambda x: -len(x[0]))

def bpe_encode(text_segment):
    """BPE-encode a segment after space→▁ normalisation (no leading ▁)."""
    normalised = ''
    for c in text_segment:
        if c == ' ':
            normalised += SP
        else:
            normalised += c
    enc = normalised.encode('utf-8')
    ids = []
    i = 0
    while i < len(enc):
        best_len = 0
        best_id = -1
        max_len = min(64, len(enc) - i)
        for l in range(max_len, 0, -1):
            sub = enc[i:i+l].decode('utf-8', errors='replace')
            if sub in rev:
                best_len = l
                best_id = rev[sub]
                break
        if best_id >= 0:
            ids.append(best_id)
            i += best_len
        else:
            b = enc[i]
            if b in byte_to_id:
                ids.append(byte_to_id[b])
            i += 1
    return ids

def encode_raw(text):
    """Split on EOS/BOS first, then BPE-encode each segment."""
    ids = []
    pos = 0
    while pos < len(text):
        # Check for special token at current position
        matched = False
        for s, sid in SPECIALS:
            if text[pos:pos+len(s)] == s:
                ids.append(sid)
                pos += len(s)
                matched = True
                break
        if matched:
            continue
        # Accumulate regular text up to next special token
        seg = ''
        while pos < len(text):
            at_special = any(text[pos:pos+len(s)] == s for s, _ in SPECIALS)
            if at_special:
                break
            seg += text[pos]
            pos += 1
        if seg:
            ids.extend(bpe_encode(seg))
    return ids

# Build the Zephyr chat prompt
sys_msg = "You are a helpful assistant."
user_msg = "What is 2+2?"

lt = chr(60)
gt = chr(62)
pipe = chr(124)
sys_tag   = lt + pipe + "system"    + pipe + gt
user_tag  = lt + pipe + "user"      + pipe + gt
asst_tag  = lt + pipe + "assistant" + pipe + gt
eos_str   = lt + "/s" + gt

prompt = (
    sys_tag + "\n" + sys_msg + eos_str + "\n" +
    user_tag + "\n" + user_msg + eos_str + "\n" +
    asst_tag + "\n"
)

print("=== Prompt string ===")
print(repr(prompt))
print()

ids = encode_raw(prompt)
# BOS prepended by llama_generate
full = [1] + ids

print(f"Total tokens (with BOS): {len(full)}")
print()
print("=== Token sequence (id: vocab_string) ===")
for i, tid in enumerate(full):
    s = toks[tid] if tid < len(toks) else '???'
    print(f"  [{i:3d}] id={tid:5d}  {repr(s)}")
