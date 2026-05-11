#!/usr/bin/env python3
"""Check special/control token entries in the GGUF vocabulary."""
import sys
sys.path.insert(0, 'llama.cpp/gguf-py')
from gguf import GGUFReader, GGUFValueType

r = GGUFReader('models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')

toks = None
types = None
for f in r.fields.values():
    if f.name == 'tokenizer.ggml.tokens':
        toks = [bytes(f.parts[p]).decode('utf-8', errors='replace') for p in f.data]
    if f.name == 'tokenizer.ggml.token_type':
        types = [int(f.parts[p][0]) for p in f.data]

# token_type: 1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte
print("=== Special/control tokens (type != 1 and type != 6) ===")
for i, (t, ty) in enumerate(zip(toks, types)):
    if ty not in (1, 6):
        print(f"  id={i:5d}  type={ty}  repr={repr(t)}")

print()
print("=== Tokens containing '<|' ===")
for i, t in enumerate(toks):
    if '<|' in t or '|>' in t:
        ty = types[i] if types else '?'
        print(f"  id={i:5d}  type={ty}  repr={repr(t)}")
