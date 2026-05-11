#!/usr/bin/env python3
"""Compare greedy longest-match vs true BPE tokenization for the chat prompt."""
import sys
sys.path.insert(0, 'llama.cpp/gguf-py')
from gguf import GGUFReader

r = GGUFReader('models/tinyllama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')

toks, merges_raw = None, None
for f in r.fields.values():
    if f.name == 'tokenizer.ggml.tokens':
        toks = [bytes(f.parts[p]).decode('utf-8', errors='replace') for p in f.data]
    if f.name == 'tokenizer.ggml.merges':
        merges_raw = [bytes(f.parts[p]).decode('utf-8', errors='replace') for p in f.data]

rev = {t: i for i, t in enumerate(toks)}
SP = '\u2581'
BOS_STR, EOS_STR = toks[1], toks[2]
SPECIALS = [(BOS_STR, 1), (EOS_STR, 2)]

# Build merge priority dict: (a, b) -> rank
merge_rank = {}
for rank, m in enumerate(merges_raw):
    parts = m.split(' ', 1)
    if len(parts) == 2:
        merge_rank[(parts[0], parts[1])] = rank


def bpe_tokenize(text):
    """True BPE tokenization: start with chars, apply merges in order."""
    # Replace spaces with SP marker
    normalized = ''.join(SP if c == ' ' else c for c in text)

    # Start with individual UTF-8 characters (as strings)
    # But we need to match what llama.cpp does: start with individual BYTES
    # encoded as byte-fallback tokens, then merge.
    # Actually llama BPE starts with individual chars (each char is a token).
    tokens = list(normalized)

    # Iteratively apply the best (lowest-rank) merge
    while True:
        best_rank = float('inf')
        best_i = -1
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            r = merge_rank.get(pair, float('inf'))
            if r < best_rank:
                best_rank = r
                best_i = i
        if best_i == -1 or best_rank == float('inf'):
            break
        # Apply merge
        merged = tokens[best_i] + tokens[best_i+1]
        tokens = tokens[:best_i] + [merged] + tokens[best_i+2:]

    # Convert to IDs
    ids = []
    for t in tokens:
        if t in rev:
            ids.append(rev[t])
        else:
            # byte fallback
            for b in t.encode('utf-8'):
                key = '<0x%02X>' % b
                ids.append(rev.get(key, -1))
    return ids


def encode_raw_greedy(text):
    """Greedy longest-match with special-token pre-splitting (current C++ approach)."""
    byte_to_id = {}
    for b in range(256):
        key = '<0x%02X>' % b
        if key in rev:
            byte_to_id[b] = rev[key]

    def bpe_seg(seg):
        normalised = ''.join(SP if c == ' ' else c for c in seg)
        enc = normalised.encode('utf-8')
        ids = []
        i = 0
        while i < len(enc):
            best_id, best_len = -1, 0
            for l in range(min(64, len(enc)-i), 0, -1):
                sub = enc[i:i+l].decode('utf-8', errors='replace')
                if sub in rev:
                    best_id, best_len = rev[sub], l
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

    ids, pos = [], 0
    while pos < len(text):
        matched = False
        for s, sid in SPECIALS:
            if text[pos:pos+len(s)] == s:
                ids.append(sid)
                pos += len(s)
                matched = True
                break
        if matched:
            continue
        seg = ''
        while pos < len(text):
            if any(text[pos:pos+len(s)] == s for s, _ in SPECIALS):
                break
            seg += text[pos]
            pos += 1
        if seg:
            ids.extend(bpe_seg(seg))
    return ids


# Build chat prompt
lt, gt, pipe = chr(60), chr(62), chr(124)
sys_tag  = lt + pipe + 'system'    + pipe + gt
user_tag = lt + pipe + 'user'      + pipe + gt
asst_tag = lt + pipe + 'assistant' + pipe + gt
prompt = (
    sys_tag + '\nYou are a helpful assistant.' + EOS_STR + '\n' +
    user_tag + '\nWhat is 2+2?' + EOS_STR + '\n' +
    asst_tag + '\n'
)

# Tokenize with both methods (no BOS yet, llama_generate adds it)
def encode_bpe_parse_special(text):
    """BPE with parse_special=True: pre-split on EOS/BOS, then BPE each segment."""
    ids, pos = [], 0
    while pos < len(text):
        matched = False
        for s, sid in SPECIALS:
            if text[pos:pos+len(s)] == s:
                ids.append(sid)
                pos += len(s)
                matched = True
                break
        if matched:
            continue
        seg = ''
        while pos < len(text):
            if any(text[pos:pos+len(s)] == s for s, _ in SPECIALS):
                break
            seg += text[pos]
            pos += 1
        if seg:
            ids.extend(bpe_tokenize(seg))
    return ids

bpe_ids    = encode_bpe_parse_special(prompt)
greedy_ids = encode_raw_greedy(prompt)

print(f"{'idx':>4}  {'BPE id':>7} {'BPE tok':<16}  {'Greedy id':>9} {'Greedy tok':<16}  {'match'}")
print('-' * 72)
for i in range(max(len(bpe_ids), len(greedy_ids))):
    b_id  = bpe_ids[i]   if i < len(bpe_ids)    else -1
    g_id  = greedy_ids[i] if i < len(greedy_ids) else -1
    b_tok = repr(toks[b_id])  if 0 <= b_id < len(toks)  else '---'
    g_tok = repr(toks[g_id])  if 0 <= g_id < len(toks)  else '---'
    match = 'OK' if b_id == g_id else '*** DIFF ***'
    print(f"{i:4d}  {b_id:7d} {b_tok:<16}  {g_id:9d} {g_tok:<16}  {match}")

print(f"\nBPE total: {len(bpe_ids)}, Greedy total: {len(greedy_ids)}")
