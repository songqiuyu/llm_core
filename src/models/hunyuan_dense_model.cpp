#include "axonforge/models/hunyuan_dense.hpp"
#include "axonforge/models/llama.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace axonforge {

std::vector<int32_t> hunyuan_dense_generate(
    const Engine& engine,
    std::vector<int32_t> prompt_ids,
    const HunyuanDenseConfig& cfg) {
    LlamaConfig lc;
    lc.max_new_tokens     = cfg.max_new_tokens;
    lc.temperature        = cfg.temperature;
    lc.top_k              = cfg.top_k;
    lc.verbose            = cfg.verbose;
    lc.n_threads          = cfg.n_threads;
    lc.max_context_len    = cfg.max_context_len;
    lc.top_p              = cfg.top_p;
    lc.rep_penalty        = cfg.rep_penalty;
    lc.rep_penalty_last_n = cfg.rep_penalty_last_n;
    lc.on_token           = cfg.on_token;
    return llama_generate(engine, std::move(prompt_ids), lc);
}

namespace {

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
        return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
    }
};

static std::array<std::string, 256> build_byte_encoder() {
    std::array<std::string, 256> enc{};
    auto append_utf8 = [](std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    };

    std::vector<uint32_t> bs;
    for (uint32_t b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (uint32_t b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (uint32_t b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);

    std::vector<uint32_t> cs = bs;
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }
    for (size_t i = 0; i < bs.size(); ++i)
        append_utf8(enc[bs[i]], cs[i]);
    return enc;
}

static const std::array<std::string, 256>& byte_encoder() {
    static const auto enc = build_byte_encoder();
    return enc;
}

static uint32_t next_cp(std::string_view s, size_t& i) {
    const uint8_t c = static_cast<uint8_t>(s[i]);
    if (c < 0x80) return s[i++];
    uint32_t cp = 0;
    int n = 0;
    if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; n = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; n = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; n = 4; }
    else { ++i; return 0xFFFD; }
    ++i;
    for (int k = 1; k < n && i < s.size(); ++k, ++i)
        cp = (cp << 6) | (static_cast<uint8_t>(s[i]) & 0x3F);
    return cp;
}

static std::string cp_utf8(uint32_t cp) {
    std::string out;
    if (cp < 0x80) out.push_back(static_cast<char>(cp));
    else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

static bool is_ascii_letter(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}
static bool is_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }
static bool is_space(uint32_t cp) { return cp == ' ' || cp == '\t' || cp == '\r' || cp == '\n'; }
static bool is_cjk_or_kana(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3040 && cp <= 0x30FF) ||
           (cp >= 0x3400 && cp <= 0x4DBF);
}

static std::vector<std::string> hunyuan_dense_pretokenize(std::string_view text) {
    std::vector<std::string> cps;
    for (size_t i = 0; i < text.size(); ) cps.push_back(cp_utf8(next_cp(text, i)));

    auto cp_at = [&](size_t i) -> uint32_t {
        if (i >= cps.size()) return 0xFFFFFFFFu;
        size_t p = 0;
        return next_cp(cps[i], p);
    };

    std::vector<std::string> out;
    for (size_t i = 0; i < cps.size(); ) {
        const uint32_t cp = cp_at(i);
        std::string seg;
        if (is_digit(cp)) {
            int n = 0;
            while (i < cps.size() && is_digit(cp_at(i)) && n < 3) {
                seg += cps[i++];
                ++n;
            }
            out.push_back(seg);
            continue;
        }
        if (is_cjk_or_kana(cp)) {
            while (i < cps.size() && is_cjk_or_kana(cp_at(i))) seg += cps[i++];
            out.push_back(seg);
            continue;
        }
        if (cp == '\r' || cp == '\n') {
            while (i < cps.size() && (cp_at(i) == '\r' || cp_at(i) == '\n')) seg += cps[i++];
            out.push_back(seg);
            continue;
        }
        if (is_space(cp)) {
            while (i < cps.size() && is_space(cp_at(i)) && cp_at(i) != '\r' && cp_at(i) != '\n')
                seg += cps[i++];
            out.push_back(seg);
            continue;
        }
        if (is_ascii_letter(cp) || (cp > 0x7F && !is_cjk_or_kana(cp) && !is_space(cp))) {
            while (i < cps.size()) {
                const uint32_t c = cp_at(i);
                if (is_digit(c) || is_space(c) || is_cjk_or_kana(c)) break;
                if (c < 0x80 && !is_ascii_letter(c)) break;
                seg += cps[i++];
            }
            out.push_back(seg);
            continue;
        }
        while (i < cps.size()) {
            const uint32_t c = cp_at(i);
            if (is_digit(c) || is_space(c) || is_cjk_or_kana(c) || is_ascii_letter(c)) break;
            seg += cps[i++];
        }
        if (!seg.empty()) out.push_back(seg);
    }
    return out;
}

static bool token_is_special(const std::string& tok, int32_t type) {
    return type == 3 || type == 4 ||
           (tok.size() >= 4 && tok[0] == '<' && tok[1] == '|') ||
           tok.rfind("<\xEF\xBD\x9C" "hy_", 0) == 0 ||
           tok.rfind("<\xEF\xBD\x9C" "human", 0) == 0 ||
           tok.rfind("<\xEF\xBD\x9C" "bot", 0) == 0;
}

} // namespace

std::vector<int32_t> hunyuan_dense_encode(const Engine& engine,
                                          std::string_view text,
                                          bool raw) {
    const auto& vocab = engine.vocabulary();
    const auto& merges = engine.tokenizer_merges();
    if (vocab.empty())
        throw std::runtime_error("hunyuan_dense_encode: vocabulary is empty");
    if (merges.empty())
        throw std::runtime_error("hunyuan_dense_encode: tokenizer.ggml.merges is empty");

    std::unordered_map<std::string, int32_t> token_to_id;
    token_to_id.reserve(vocab.size());
    for (int32_t i = 0; i < static_cast<int32_t>(vocab.size()); ++i)
        token_to_id.emplace(vocab[i], i);

    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> rank;
    rank.reserve(merges.size());
    for (int i = 0; i < static_cast<int>(merges.size()); ++i) {
        const std::string& m = merges[i];
        const size_t sp = m.find(' ');
        if (sp != std::string::npos)
            rank.emplace(std::make_pair(m.substr(0, sp), m.substr(sp + 1)), i);
    }

    const auto& token_types = engine.tokenizer_token_types();
    std::vector<std::pair<std::string, int32_t>> specials;
    if (raw) {
        for (int32_t i = 0; i < static_cast<int32_t>(vocab.size()); ++i) {
            const int32_t type = i < static_cast<int32_t>(token_types.size()) ? token_types[i] : 0;
            if (token_is_special(vocab[i], type)) specials.push_back({vocab[i], i});
        }
    }

    auto bpe_segment = [&](std::string_view seg, std::vector<int32_t>& ids) {
        const auto& enc = byte_encoder();
        for (const std::string& piece : hunyuan_dense_pretokenize(seg)) {
            std::vector<std::string> syms;
            for (uint8_t b : piece) syms.push_back(enc[b]);
            while (syms.size() > 1) {
                int best = std::numeric_limits<int>::max();
                size_t best_i = syms.size();
                for (size_t i = 0; i + 1 < syms.size(); ++i) {
                    auto it = rank.find({syms[i], syms[i + 1]});
                    if (it != rank.end() && it->second < best) {
                        best = it->second;
                        best_i = i;
                    }
                }
                if (best_i == syms.size()) break;
                syms[best_i] += syms[best_i + 1];
                syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(best_i + 1));
            }
            for (const std::string& s : syms) {
                auto it = token_to_id.find(s);
                if (it != token_to_id.end()) {
                    ids.push_back(it->second);
                } else {
                    for (uint8_t b : s) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
                        auto bit = token_to_id.find(buf);
                        if (bit != token_to_id.end()) ids.push_back(bit->second);
                    }
                }
            }
        }
    };

    std::vector<int32_t> ids;
    size_t pos = 0;
    while (pos < text.size()) {
        bool matched = false;
        if (raw) {
            const std::pair<std::string, int32_t>* best = nullptr;
            for (const auto& sp : specials) {
                if (sp.first.size() <= text.size() - pos &&
                    text.substr(pos, sp.first.size()) == sp.first) {
                    if (!best || sp.first.size() > best->first.size())
                        best = &sp;
                }
            }
            if (best) {
                ids.push_back(best->second);
                pos += best->first.size();
                matched = true;
            }
        }
        if (matched) continue;

        size_t next = text.size();
        if (raw) {
            for (const auto& sp : specials) {
                const size_t p = std::string_view(text).find(sp.first, pos);
                if (p != std::string_view::npos) next = std::min(next, p);
            }
        }
        bpe_segment(text.substr(pos, next - pos), ids);
        pos = next;
    }
    return ids;
}

std::string hunyuan_dense_decode(const Engine& engine,
                                 const std::vector<int32_t>& token_ids) {
    const auto& vocab = engine.vocabulary();
    const auto& types = engine.tokenizer_token_types();
    const auto& enc = byte_encoder();
    std::unordered_map<std::string, char> dec;
    for (int b = 0; b < 256; ++b) dec.emplace(enc[b], static_cast<char>(b));

    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };

    std::string out;
    for (int32_t id : token_ids) {
        if (id < 0 || id >= static_cast<int32_t>(vocab.size())) continue;
        const int32_t type = id < static_cast<int32_t>(types.size()) ? types[id] : 0;
        const std::string& tok = vocab[id];
        if (token_is_special(tok, type)) continue;
        if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>') {
            const int hi = hex(tok[3]), lo = hex(tok[4]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                continue;
            }
        }

        for (size_t i = 0; i < tok.size(); ) {
            size_t start = i;
            (void)next_cp(tok, i);
            std::string cp(tok.substr(start, i - start));
            auto it = dec.find(cp);
            if (it != dec.end()) out.push_back(it->second);
            else out += cp;
        }
    }
    return out;
}

} // namespace axonforge
