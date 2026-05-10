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
//   -b <id>           Backend (default: cpu_x86)
//   --list-backends   Print available backends and exit
//   --version         Print version and exit
//
// Interactive commands (type at the prompt):
//   /help             Show help
//   /clear            Reset turn counter display
//   /quit  or  /exit  Quit
//   Ctrl-D            Quit (EOF)
// ============================================================

#include "axonforge/engine.hpp"
#include "axonforge/backend.hpp"
#include "axonforge/models/gpt2.hpp"
#include "axonforge/models/llama.hpp"
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

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "AxonForge Inference CLI v%d.%d.%d\n\n"
        "Usage:\n"
        "  Single-shot:   %s -m <model.gguf> -p <prompt> [options]\n"
        "  Interactive:   %s -m <model.gguf> -i [--prefix <text>] [options]\n\n"
        "Options:\n"
        "  -m <path>         GGUF model file (required)\n"
        "  -p <text>         Prompt (single-shot)\n"
        "  -i, --interactive Interactive REPL mode\n"
        "  --prefix <text>   Text prepended to every user input (interactive)\n"
        "  -n <int>          Max new tokens per turn (default: 256)\n"
        "  -t <float>        Temperature (default: 0.8)\n"
        "  --top-k <int>     Top-K (default: 40)\n"
        "  --top-p <float>   Top-P (default: 0.95)\n"
        "  -b <id>           Backend (default: cpu_x86)\n"
        "  --list-backends   Print available backends\n"
        "  --version         Print version\n",
        AXONFORGE_VERSION_MAJOR,
        AXONFORGE_VERSION_MINOR,
        AXONFORGE_VERSION_PATCH,
        prog, prog);
}

// ── Shared single-turn generation ────────────────────────────────────────────

// Arch-agnostic streaming token callback type
using TokenCallback = std::function<void(int32_t)>;

// Dispatch generate to the correct model implementation based on arch
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

static void run_once(const axonforge::Engine& engine,
                     const std::string& prompt_text,
                     int max_new_tokens, float temperature, int top_k,
                     bool show_stats, bool colours) {
    using Clock = std::chrono::steady_clock;

    auto prompt_ids = engine.encode(prompt_text, /*add_bos=*/false);

    auto t_start          = Clock::now();
    bool first_token_done = false;
    double ttft_ms        = 0.0;
    int    n_gen          = 0;

    if (colours) std::printf("%s", COL_REPLY);

    dispatch_generate(engine, prompt_ids, max_new_tokens, temperature, top_k,
        [&](int32_t tok_id) {
            if (!first_token_done) {
                ttft_ms = std::chrono::duration<double, std::milli>(
                              Clock::now() - t_start).count();
                first_token_done = true;
            }
            n_gen++;
            std::string frag = engine.decode({&tok_id, 1});
            std::printf("%s", frag.c_str());
            std::fflush(stdout);
        });

    auto t_end = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(
                          t_end - t_start).count();

    if (colours) std::printf("%s", COL_RESET);
    std::printf("\n");

    if (show_stats) {
        if (colours) std::fprintf(stderr, "%s", COL_STATS);
        std::fprintf(stderr,
            "[stats] TTFT: %.0f ms | %d tokens | %.2f tok/s\n",
            ttft_ms, n_gen,
            n_gen > 0 ? n_gen / (total_ms / 1000.0) : 0.0);
        if (colours) std::fprintf(stderr, "%s", COL_RESET);
    }
}

// ── Interactive REPL ─────────────────────────────────────────────────────────

static void run_interactive(const axonforge::Engine& engine,
                             const std::string& prefix,
                             int max_new_tokens, float temperature, int top_k) {
    const bool colours = is_tty();

    if (colours) std::printf("%s", COL_BANNER);
    std::printf("AxonForge Interactive Mode  (type /help for commands, Ctrl-D to quit)\n");
    if (!prefix.empty())
        std::printf("Prefix: \"%s\"\n", prefix.c_str());
    if (colours) std::printf("%s\n", COL_RESET);
    else         std::printf("\n");

    int turn = 0;
    std::string line;

    for (;;) {
        // ── Print prompt ────────────────────────────────────────────────────
        if (colours) std::printf("%s", COL_PROMPT);
        std::printf(">>> ");
        if (colours) std::printf("%s", COL_RESET);
        std::fflush(stdout);

        // ── Read line from stdin ─────────────────────────────────────────────
        line.clear();
        int ch;
        while ((ch = std::getchar()) != EOF && ch != '\n')
            line += static_cast<char>(ch);

        if (ch == EOF) {
            std::printf("\n");
            break;  // Ctrl-D
        }

        // ── Strip leading/trailing whitespace ───────────────────────────────
        size_t s = line.find_first_not_of(" \t\r");
        size_t e = line.find_last_not_of(" \t\r");
        if (s == std::string::npos) continue;  // empty line
        line = line.substr(s, e - s + 1);

        // ── Built-in commands ────────────────────────────────────────────────
        if (line == "/quit" || line == "/exit") break;

        if (line == "/help") {
            std::printf(
                "Commands:\n"
                "  /help       Show this message\n"
                "  /clear      Clear screen\n"
                "  /quit       Exit (also Ctrl-D)\n"
                "  /exit       Exit\n"
                "Anything else is sent to the model as a prompt.\n\n");
            continue;
        }

        if (line == "/clear") {
            std::printf("\033[2J\033[H");  // ANSI clear screen + cursor home
            std::fflush(stdout);
            continue;
        }

        // ── Build full prompt and generate ───────────────────────────────────
        turn++;
        const std::string full_prompt = prefix.empty() ? line : prefix + line;

        if (colours) std::printf("%s", COL_RESET);
        run_once(engine, full_prompt, max_new_tokens, temperature, top_k,
                 /*show_stats=*/true, colours);
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
            run_interactive(engine, prefix, max_new_tokens, temperature, top_k);
        } else {
            std::fprintf(stderr, "[AxonForge] Prompt: %s\n", prompt.c_str());
            const bool colours = is_tty();
            run_once(engine, prompt, max_new_tokens, temperature, top_k,
                     /*show_stats=*/true, colours);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[AxonForge] Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
