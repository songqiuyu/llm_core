#!/bin/bash
# Test llama.cpp with TinyLlama Zephyr chat format using --chat-template
MODEL="models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
BIN="llama.cpp/build/bin/llama-cli"

echo "=== llama.cpp TinyLlama Zephyr chat ==="
"$BIN" -m "$MODEL" \
  --chat-template zephyr \
  --system "You are a helpful assistant." \
  -p "What is 2+2?" \
  -n 30 \
  --temp 0.0 \
  --no-display-prompt \
  2>/dev/null
echo ""

echo "=== llama.cpp top-5 logits (greedy) ==="
# Use --logits-all to dump logits - not available in all builds
# Instead just compare output text
"$BIN" -m "$MODEL" \
  --chat-template zephyr \
  --system "" \
  -p "What is 2+2?" \
  -n 30 \
  --temp 0.0 \
  --no-display-prompt \
  2>/dev/null

