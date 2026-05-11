#!/usr/bin/env python3
"""
Use llama-cpp-python to get exact logits for a prompt and compare
with AxonForge's top-5 output.
"""
import sys
try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python not installed, trying transformers...")
    sys.exit(1)

MODEL = "models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
PROMPT = "Once upon a time in a land"

llm = Llama(model_path=MODEL, n_ctx=512, logits_all=False, verbose=False)

# Tokenize prompt (with BOS)
tokens = llm.tokenize(PROMPT.encode(), add_bos=True)
print(f"Tokens ({len(tokens)}): {tokens}")

# Run one forward pass to get next-token logits
output = llm.eval(tokens)
logits = llm.eval_logits[-1] if hasattr(llm, 'eval_logits') else None
if logits is None:
    # Alternative: use create_completion with logprobs
    result = llm.create_completion(PROMPT, max_tokens=1, logprobs=5, temperature=0.0)
    print("Top-5 via logprobs:")
    for tok, lp in result['choices'][0]['logprobs']['top_logprobs'][0].items():
        print(f"  {repr(tok)}: logprob={lp:.3f}")
