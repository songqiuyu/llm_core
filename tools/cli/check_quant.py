#!/usr/bin/env python3
"""Check quantization types of each tensor in TinyLlama GGUF."""
import sys
sys.path.insert(0, 'llama.cpp/gguf-py')
from gguf import GGUFReader, GGMLQuantizationType

r = GGUFReader('models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')

from collections import Counter
type_counts = Counter()
for t in r.tensors:
    type_counts[t.tensor_type.name] += 1
    
print("Tensor type distribution:")
for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

print("\nFirst 15 tensors:")
for t in list(r.tensors)[:15]:
    print(f"  {t.name:<50} {t.tensor_type.name}")
