#!/usr/bin/env python3
import sys
sys.path.insert(0, 'llama.cpp/gguf-py')
from gguf import GGUFReader

r = GGUFReader('models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
toks = None
for f in r.fields.values():
    if f.name == 'tokenizer.ggml.tokens':
        toks = [bytes(f.parts[p]).decode('utf-8', errors='replace') for p in f.data]
        break

if not toks:
    print('No tokens found'); sys.exit(1)

targets = ['</s>', '<s>', '<unk>', '<|system|>', '
