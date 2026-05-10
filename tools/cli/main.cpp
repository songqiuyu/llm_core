// ============================================================
// axonforge-cli — command-line inference tool
//
// Modes:
//   Single-shot:   axonforge-cli -m <model.gguf> -p <prompt> [options]
//   Interactive:   axonforge-cli -m <model.gguf> -i [--prefix <text>] [options]
//
// Options:
//   -m <path>         GGUF model file path (required)
//   -p <text>         Prompt (single-shot mode)
//   -i / --interactive  Enter interactive REPL mode
//   --prefix <text>   Text prepended to every user input (interactive only)
//   -n <int>          Max new tokens per turn (default: 256)
//   -t <float>        Temperature (default: 0.8)
//   --top-k <int>     Top-K (default: 40)
//   --top-p <float>   Top-P (default: 0.95)
//   -V / --verbose    Show rich inference stats (throughput breakdown, RSS memory)
//   -b <id>           Backend (default: cpu_x86)
//   --list-backends   Print available backends and exit
//   --version         Print version and exit
//
// Interactive commands (type at the prompt):
//   /help             Show help
//   /clear            Clear screen
//   /new              Reset conversation history (start a fresh topic)
//   /quit  or  /exit  Quit
//   Ctrl-D            Quit (EOF)
// ============================================================

#include "axonforge/engine.hpp"
#include "axonforge/backend.hpp"
#include "axonforge/models/gpt2.hpp"
#include "axonforge/models/llama.hpp"
#include "axonforge/dtype.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ANSI colour codes (disabled automatically when stdout is not a tty)
#ifdef _WIN32
static bool is_tty() { return false; }
#else
#include <unistd.h>
static bool is_tty() { return isatty(STDOUT_FILENO); }
#endif

static const char* COL_PROMPT  = "\033[1;32m";  // bold green
static const char* COL_REPLY   = "\033[0;37m";  // light grey
static const char* COL_STATS   = "\033[0;36m";  // cyan
static const char* COL_RESET   = "\033[0m";
static const char* COL_BANNER  = "\033[1;33m";  // bold yellow

// ── RSS memory (Linux) ────────────────────────────────────────────────────────
static long get_rss_mb() noexcept {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    long vmrss = -1;
    char buf[256];
    while (std::fgets(buf, sizeof(buf), f))
        if (std::sscanf(buf, "VmRSS: %ld kB", &vmrss) == 1) break;
    std::fclose(f);
    return vmrss > 0 ? vmrss / 1024 : -1;
}

// ── ASCII/Unicode banner ──────────────────────────────────────────────────────
static void print_banner(bool colours) noexcept {
    if (colours) std::printf("\033[1;36m");
    std::printf(
        "\n"
        "  \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97 \xe2\x96\x88\xe2\x96\x88\xe2\x95\x97  \xe2\x96\x88\xe2\x96\x88\xe2\x95\x97 \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97 \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97   \xe2\x96\x88\xe2\x96\x88\xe2\x95\x97\n"
        " \xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97\xe2\x95\x9a\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x9d\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97  \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91\n"
        " \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x91 \xe2\x95\x9a\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x9d \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91   \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97 \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91\n"
        " \xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x96\x88\xe2\x96\x88\xe2\x95\x91 \xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97 \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91   \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91\xe2\x96\x88\xe2\x96\x88\xe2\x95\x91\xe2\x95\x9a\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97\xe2\x96\x88\xe2\x96\x88\xe2\x95\x91\n"
        " \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91  \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x9d \xe2\x96\x88\xe2\x96\x88\xe2\x95\x97\xe2\x95\x9a\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x9d\xe2\x96\x88\xe2\x96\x88\xe2\x95\x91 \xe2\x95\x9a\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x91\n"
        " \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x9d  \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x9d\xe2\x95\x9a\xe2\x95\x90\xe2\x95\x9d  \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x9d \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x9d  \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d\n"
    );
    if (colours) std::printf("\033[1;35m");
    std::printf(
        " \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97 \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97 \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97  \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97 \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97\n"
        " \xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d \xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d\n"
        " \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97  \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91   \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x9d\xe2\x96\x88\xe2\x96\x88\xe2\x95\x91  \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97  \n"
        " \xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d  \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91   \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97\xe2\x96\x88\xe2\x96\x88\xe2\x95\x91   \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d  \n"
        " \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91     \xe2\x95\x9a\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x9d\xe2\x96\x88\xe2\x96\x88\xe2\x95\x91  \xe2\x96\x88\xe2\x96\x88\xe2\x95\x91\xe2\x95\x9a\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x9d\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x95\x97\n"
        " \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x9d      \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x9d  \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x9d \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d\n"
    );
    if (colours) std::printf("\033[0;37m");
    std::printf("                          \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 ChiourainSoong \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\n\n");
    if (colours) std::printf("\033[0m");
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "AxonForge Inference CLI v%d.%d.%d\n\n"
        "Usage:\n"
        "  Single-shot:   %s -m <model.gguf> -p <prompt> [options]\n"
        "  Interactive:   %s -m <model.gguf> -i [--prefix <text>] [options]\n\n"
        "Options:\n"
        "  -m <path>         GGUF model file (required)\n"
        "  -p <text>         Prompt (single-shot)\n"
        "  -i, --interactive Interactive REPL mode (multi-turn conversation)\n"
        "  --prefix <text>   Text prepended to every user input (interactive)\n"
        "  -n <int>          Max new tokens per turn (default: 256)\n"
        "  -t <float>        Temperature (default: 0.8)\n"
        "  --top-k <int>     Top-K (default: 40)\n"
        "  --top-p <float>   Top-P (default: 0.95)\n"
        "  -V, --verbose     Rich inference stats (throughput, RSS memory)\n"
        "  -b <id>           Backend (default: cpu_x86)\n"
        "  --list-backends   Print available backends\n"
        "  --version         Print version\n",
        AXONFORGE_VERSION_MAJOR,
        AXONFORGE_VERSION_MINOR,
        AXONFORGE_VERSION_PATCH,
        prog, prog);
}

// ── Shared generation core ────────────────────────────────────────────────────

using TokenCallback = std::function<void(int32_t)>;

static void dispatch_generate(const axonforge::Engine& engine,
                               const std::vector<int32_t>& prompt_ids,
                               int max_new_tokens, float temperature, int top_k,
                               TokenCallback on_token) {
    const std::string& arch = engine.model_config().arch;
    if (arch == "gpt2") {
        axonforge::Gpt2Config cfg;
        cfg.max_new_tokens = max_new_tokens;
        cfg.temperature    = temperature;
        cfg.top_k          = top_k;
        cfg.on_token       = std::move(on_token);
        axonforge::gpt2_generate(engine, prompt_ids, cfg);
    } else {
        axonforge::LlamaConfig cfg;
        cfg.max_new_tokens = max_new_tokens;
        cfg.temperature    = temperature;
        cfg.top_k          = top_k;
        cfg.on_token       = std::move(on_token);
        axonforge::llama_generate(engine, prompt_ids, cfg);
    }
}

// Core function: stream tokens, print them, return generated IDs + stats.
static std::vector<int32_t>
generate_and_print(const axonforge::Engine& engine,
                   const std::vector<int32_t>& prompt_ids,
                   int max_new_tokens, float temperature, int top_k,
                   bool show_stats, bool verbose, bool colours) {
    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();
    bool first_done = false;
    double ttft_ms  = 0.0;

    std::vector<int32_t> gen_ids;
    gen_ids.reserve(max_new_tokens);

    if (colours) std::printf("%s", COL_REPLY);

    dispatch_generate(engine, prompt_ids, max_new_tokens, temperature, top_k,
        [&](int32_t tok_id) {
            if (!first_done) {
                ttft_ms = std::chrono::duration<double, std::milli>(
                              Clock::now() - t_start).count();
                first_done = true;
            }
            gen_ids.push_back(tok_id);
            const std::string frag = engine.decode({&tok_id, 1});
            std::printf("%s", frag.c_str());
            std::fflush(stdout);
        });

    const double total_ms = std::chrono::duration<double, std::milli>(
                                Clock::now() - t_start).count();
    const int n_gen    = (int)gen_ids.size();
    const int n_prompt = (int)prompt_ids.size();

    if (colours) std::printf("%s", COL_RESET);
    std::printf("\n");

    if (show_stats) {
        if (colours) std::fprintf(stderr, "%s", COL_STATS);
        if (verbose) {
            const double decode_ms   = total_ms - ttft_ms;
            const double prefill_tps = (n_prompt > 0 && ttft_ms > 0.1)
                ? n_prompt / (ttft_ms  / 1000.0) : 0.0;
            const double decode_tps  = (n_gen > 1 && decode_ms > 1.0)
                ? (n_gen - 1) / (decode_ms / 1000.0) : 0.0;
            const long rss = get_rss_mb();
            std::fprintf(stderr,
                "\xe2\x94\x8c\xe2\x94\x80 Inference Stats \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\n"
                "\xe2\x94\x82  Context tokens  : %-6d                   \xe2\x94\x82\n"
                "\xe2\x94\x82  Generated       : %-6d                   \xe2\x94\x82\n"
                "\xe2\x94\x82  TTFT            : %8.1f ms              \xe2\x94\x82\n"
                "\xe2\x94\x82  Prefill speed   : %8.1f tok/s           \xe2\x94\x82\n"
                "\xe2\x94\x82  Decode speed    : %8.2f tok/s           \xe2\x94\x82\n"
                "\xe2\x94\x82  Total time      : %8.1f ms              \xe2\x94\x82\n"
                "\xe2\x94\x82  Memory (RSS)    : %5ld MB                 \xe2\x94\x82\n"
                "\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\n",
                n_prompt, n_gen, ttft_ms, prefill_tps, decode_tps, total_ms, rss);
        } else {
            std::fprintf(stderr,
                "[stats] TTFT: %.0f ms | %d tokens | %.2f tok/s\n",
                ttft_ms, n_gen,
                n_gen > 0 ? n_gen / (total_ms / 1000.0) : 0.0);
        }
        if (colours) std::fprintf(stderr, "%s", COL_RESET);
    }
    return gen_ids;
}

static void run_once(const axonforge::Engine& engine,
                     const std::string& prompt_text,
                     int max_new_tokens, float temperature, int top_k,
                     bool show_stats, bool verbose, bool colours) {
    const auto prompt_ids = engine.encode(prompt_text, /*add_bos=*/false);
    generate_and_print(engine, prompt_ids, max_new_tokens, temperature, top_k,
                       show_stats, verbose, colours);
}

// ── Interactive REPL ─────────────────────────────────────────────────────────

static void run_interactive(const axonforge::Engine& engine,
                             const std::string& prefix,
                             int max_new_tokens, float temperature, int top_k,
                             bool verbose) {
    const bool colours = is_tty();
    print_banner(colours);

    // ── Model info line ──────────────────────────────────────────────────────
    const auto& mc = engine.model_config();
    if (colours) std::printf("%s", COL_BANNER);
    std::printf("  Model : %s | %d layers | %d heads (KV:%d) | hidden %d | vocab %d | %s\n",
        mc.arch.c_str(), mc.n_layers, mc.n_heads, mc.n_kv_heads,
        mc.hidden_dim, mc.vocab_size,
        std::string(axonforge::dtype_name(mc.weight_dtype)).c_str());
    if (!prefix.empty())
        std::printf("  Prefix: \"%s\"\n", prefix.c_str());
    std::printf("  Type /help for commands, /new to reset history, Ctrl-D to quit\n");
    if (colours) std::printf("%s", COL_RESET);
    std::printf("\n");

    // ── Conversation state ───────────────────────────────────────────────────
    // history_ids accumulates all past token IDs (without BOS; llama_generate
    // prepends BOS automatically). Each turn: prompt = history + user_ids.
    std::vector<int32_t> history_ids;
    const int max_context  = mc.max_seq_len > 0 ? std::min(mc.max_seq_len, 4096) : 4096;
    const int history_cap  = max_context - max_new_tokens - 16;  // leave room for decode

    std::string line;
    for (;;) {
        // ── Print prompt ─────────────────────────────────────────────────────
        const int ctx_used = (int)history_ids.size();
        if (colours) std::printf("%s", COL_PROMPT);
        if (verbose && ctx_used > 0)
            std::printf("[ctx:%d] >>> ", ctx_used);
        else
            std::printf(">>> ");
        if (colours) std::printf("%s", COL_RESET);
        std::fflush(stdout);

        // ── Read line ────────────────────────────────────────────────────────
        line.clear();
        int ch;
        while ((ch = std::getchar()) != EOF && ch != '\n')
            line += static_cast<char>(ch);
        if (ch == EOF) { std::printf("\n"); break; }

        // ── Strip whitespace ─────────────────────────────────────────────────
        const size_t s = line.find_first_not_of(" \t\r");
        const size_t e = line.find_last_not_of(" \t\r");
        if (s == std::string::npos) continue;
        line = line.substr(s, e - s + 1);

        // ── Built-in commands ────────────────────────────────────────────────
        if (line == "/quit" || line == "/exit") break;

        if (line == "/help") {
            std::printf(
                "Commands:\n"
                "  /help    Show this message\n"
                "  /clear   Clear screen\n"
                "  /new     Reset conversation history (start a fresh topic)\n"
                "  /quit    Exit (also Ctrl-D)\n"
                "  /exit    Exit\n"
                "Anything else is sent to the model.\n\n");
            continue;
        }
        if (line == "/clear") {
            std::printf("\033[2J\033[H");
            std::fflush(stdout);
            continue;
        }
        if (line == "/new") {
            history_ids.clear();
            if (colours) std::printf("%s", COL_STATS);
            std::printf("[Conversation history cleared. Starting fresh.]\n\n");
            if (colours) std::printf("%s", COL_RESET);
            continue;
        }

        // ── Build prompt with history ─────────────────────────────────────────
        const std::string user_text = prefix.empty() ? line : prefix + line;
        auto user_ids = engine.encode(user_text, /*add_bos=*/false);

        // Truncate oldest history tokens if context would overflow
        if ((int)(history_ids.size() + user_ids.size()) > history_cap) {
            const int excess = (int)(history_ids.size() + user_ids.size()) - history_cap;
            if (excess < (int)history_ids.size())
                history_ids.erase(history_ids.begin(), history_ids.begin() + excess);
            else
                history_ids.clear();
            if (verbose) std::fprintf(stderr, "[context window trimmed]\n");
        }

        std::vector<int32_t> prompt_ids;
        prompt_ids.reserve(history_ids.size() + user_ids.size());
        prompt_ids.insert(prompt_ids.end(), history_ids.begin(), history_ids.end());
        prompt_ids.insert(prompt_ids.end(), user_ids.begin(), user_ids.end());

        // ── Generate ─────────────────────────────────────────────────────────
        const auto gen_ids = generate_and_print(
            engine, prompt_ids, max_new_tokens, temperature, top_k,
            /*show_stats=*/true, verbose, colours);

        // ── Update history: prompt + generated response ───────────────────────
        history_ids = std::move(prompt_ids);
        history_ids.insert(history_ids.end(), gen_ids.begin(), gen_ids.end());

        std::printf("\n");
    }
    std::printf("Bye.\n");
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string prompt;
    std::string prefix;
    std::string backend_id    = "cpu_x86";
    int         max_new_tokens = 256;
    float       temperature    = 0.8f;
    int         top_k          = 40;
    float       top_p          = 0.95f;
    bool        interactive    = false;
    bool        verbose        = false;

    if (argc == 1) { print_usage(argv[0]); return 0; }

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--version") == 0) {
            std::printf("axonforge-cli %d.%d.%d\n",
                AXONFORGE_VERSION_MAJOR, AXONFORGE_VERSION_MINOR, AXONFORGE_VERSION_PATCH);
            return 0;
        }
        if (std::strcmp(arg, "--list-backends") == 0) {
            std::printf("Available backends:\n");
            for (const auto& id : axonforge::BackendRegistry::instance().available_backends())
                std::printf("  %s\n", id.c_str());
            return 0;
        }
        if ((std::strcmp(arg, "-i") == 0 || std::strcmp(arg, "--interactive") == 0))
            { interactive = true; continue; }
        if ((std::strcmp(arg, "-V") == 0 || std::strcmp(arg, "--verbose") == 0))
            { verbose = true; continue; }
        if (std::strcmp(arg, "-m") == 0 && i + 1 < argc) { model_path = argv[++i]; continue; }
        if (std::strcmp(arg, "-p") == 0 && i + 1 < argc) { prompt     = argv[++i]; continue; }
        if (std::strcmp(arg, "--prefix") == 0 && i + 1 < argc) { prefix = argv[++i]; continue; }
        if (std::strcmp(arg, "-b") == 0 && i + 1 < argc) { backend_id = argv[++i]; continue; }
        if (std::strcmp(arg, "-n") == 0 && i + 1 < argc) { max_new_tokens = std::atoi(argv[++i]); continue; }
        if (std::strcmp(arg, "-t") == 0 && i + 1 < argc) { temperature    = std::atof(argv[++i]); continue; }
        if (std::strcmp(arg, "--top-k") == 0 && i + 1 < argc) { top_k   = std::atoi(argv[++i]); continue; }
        if (std::strcmp(arg, "--top-p") == 0 && i + 1 < argc) { top_p   = std::atof(argv[++i]); continue; }
        std::fprintf(stderr, "Unknown argument: %s\n", arg);
        print_usage(argv[0]);
        return 1;
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "Error: -m <model.gguf> is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (!interactive && prompt.empty()) {
        std::fprintf(stderr, "Error: -p <prompt> is required in single-shot mode (or use -i)\n");
        print_usage(argv[0]);
        return 1;
    }

    // ── Load model ──────────────────────────────────────────────────────────
    axonforge::EngineConfig eng_cfg;
    eng_cfg.backend = backend_id;

    std::fprintf(stderr, "[AxonForge] Loading model: %s\n", model_path.c_str());
    std::fprintf(stderr, "[AxonForge] Backend: %s\n", backend_id.c_str());

    try {
        auto engine = axonforge::Engine::from_gguf(model_path, eng_cfg);

        if (interactive) {
            run_interactive(engine, prefix, max_new_tokens, temperature, top_k, verbose);
        } else {
            std::fprintf(stderr, "[AxonForge] Prompt: %s\n", prompt.c_str());
            const bool colours = is_tty();
            run_once(engine, prompt, max_new_tokens, temperature, top_k,
                     /*show_stats=*/true, verbose, colours);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[AxonForge] Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
