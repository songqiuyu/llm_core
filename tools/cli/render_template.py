#!/usr/bin/env python3
"""Render the GGUF chat template with Jinja2 to see exact output."""
try:
    from jinja2 import Environment
except ImportError:
    print("pip install jinja2")
    raise

import sys
sys.path.insert(0, 'llama.cpp/gguf-py')
from gguf import GGUFReader

r = GGUFReader('models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
tmpl_str = None
for f in r.fields.values():
    if f.name == 'tokenizer.chat_template':
        tmpl_str = b''.join(bytes(f.parts[p]) for p in f.data).decode('utf-8')
        break

env = Environment(trim_blocks=True, lstrip_blocks=True)
template = env.from_string(tmpl_str)

messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "Tell me a short joke."},
]

result = template.render(messages=messages, eos_token="</s>", add_generation_prompt=True)
print("=== Rendered prompt (repr) ===")
print(repr(result))
print()
print("=== Rendered prompt (raw) ===")
print(result)
