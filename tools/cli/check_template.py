#!/usr/bin/env python3
"""Check the chat template stored in the GGUF model."""
import sys
sys.path.insert(0, 'llama.cpp/gguf-py')
from gguf import GGUFReader

r = GGUFReader('models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
for f in r.fields.values():
    if f.name == 'tokenizer.chat_template':
        data = b''.join(bytes(f.parts[p]) for p in f.data)
        print("=== tokenizer.chat_template ===")
        print(data.decode('utf-8'))
        break
