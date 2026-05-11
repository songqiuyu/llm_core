#!/usr/bin/env python3
"""Compare our token IDs to what HuggingFace/SentencePiece produces via llama.cpp-style merge table."""
import sys
sys.path.insert(0, 'llama.cpp/gguf-py')
from gguf import GGUFReader

r = GGUFReader('models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')

toks = None
merges = None
for f in r.fields.values():
    if f.name == 'tokenizer.ggml.tokens':
        toks = [bytes(f.parts[p]).decode('utf-8', errors='replace') for p in f.data]
    if f.name == 'tokenizer.ggml.merges':
        merges = [bytes(f.parts[p]).decode('utf-8', errors='replace') for p in f.data]

print(f"Vocab size: {len(toks)}")
if merges:
    print(f"Merge rules: {len(merges)}")
    print("First 10 merges:")
    for m in merges[:10]:
        print(f"  {repr(m)}")
else:
    print("No 'tokenizer.ggml.merges' field found -- this model uses SPM, not BPE merges")

# Check if there are any scores (SPM uses scores, BPE uses merge order)
for f in r.fields.values():
    if f.name == 'tokenizer.ggml.scores':
        scores = [float(f.parts[p][0]) for p in f.data]
        print(f"\nScores field present ({len(scores)} entries). First 10:")
        for i in range(10):
            print(f"  [{i}] score={scores[i]:.4f}  tok={repr(toks[i])}")
        break

# Check tokenizer model type
for f in r.fields.values():
    if f.name == 'tokenizer.ggml.model':
        data = b''.join(bytes(f.parts[p]) for p in f.data).decode('utf-8')
        print(f"\ntokenizer.ggml.model = {repr(data)}")
