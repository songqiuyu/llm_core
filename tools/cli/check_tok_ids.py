#!/usr/bin/env python3
import sys
sys.path.insert(0, 'llama.cpp/gguf-py')
from gguf import GGUFReader
r = GGUFReader('models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
toks = None
for f in r.fields.values():
    if f.name == 'tokenizer.ggml.tokens':
        toks = [bytes(f.parts[p]).decode('utf-8', errors='replace') for p in f.data]

# IDs from top-5 debug
ids = [8241, 1576, 29903, 29902, 6028, 6113, 29989, 3492]
print("Token ID lookup:")
for i in ids:
    print(f"  {i} -> {repr(toks[i])}")

# Find '4' and '▁4'
sp = '\u2581'
for i, t in enumerate(toks):
    if t in ('4', sp+'4', '▁4'):
        print(f"  '4'-related: id={i} repr={repr(t)}")
