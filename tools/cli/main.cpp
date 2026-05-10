#include "axonforge/engine.hpp"
#include "axonforge/backend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ============================================================
// axonforge-cli — command-line inference tool
//
// Usage:
//   axonforge-cli -m <model.gguf> -p <prompt> [options]
//
// Options:
//   -m <path>         Path to GGUF model file
//   -p <text>         Prompt string
//   -n <int>          Max new tokens to generate (default: 256)
//   -t <float>        Temperature (default: 0.8)
//   --top-k <int>     Top-k (default: 40)
//   --top-p <float>   Top-p (default: 0.95)
//   -b <backend>      Backend id (default: cpu_x86)
//   --list-backends   List all registered backends and exit
//   --version         Print version and exit
// ============================================================

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "AxonForge Inference CLI v%d.%d.%d\n\n"
        "Usage: %s -m <model.gguf> -p <prompt> [options]\n\n"
        "Options:\n"
        "  -m <path>         GGUF model file path\n"
        "  -p <text>         Input prompt\n"
        "  -n <int>          Max new tokens (default: 256)\n"
        "  -t <float>        Temperature (default: 0.8)\n"
        "  --top-k <int>     Top-K (default: 40)\n"
        "  --top-p <float>   Top-P (default: 0.95)\n"
        "  -b <id>           Backend (default: cpu_x86)\n"
        "  --list-backends   Print available backends\n"
        "  --version         Print version\n",
        AXONFORGE_VERSION_MAJOR,
        AXONFORGE_VERSION_MINOR,
        AXONFORGE_VERSION_PATCH,
        prog);
}

int main(int argc, char* argv[]) {
    // ---- Argument parsing ----
    std::string model_path;
    std::string prompt;
    std::string backend_id = "cpu_x86";
    int         max_new_tokens = 256;
    float       temperature    = 0.8f;
    int         top_k          = 40;
    float       top_p          = 0.95f;

    if (argc == 1) { print_usage(argv[0]); return 0; }

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--version") == 0) {
            std::printf("axonforge-cli %d.%d.%d\n",
                AXONFORGE_VERSION_MAJOR,
                AXONFORGE_VERSION_MINOR,
                AXONFORGE_VERSION_PATCH);
            return 0;
        }
        if (std::strcmp(arg, "--list-backends") == 0) {
            std::printf("Available backends:\n");
            for (const auto& id : axonforge::BackendRegistry::instance().available_backends()) {
                std::printf("  %s\n", id.c_str());
            }
            return 0;
        }
        if (std::strcmp(arg, "-m") == 0 && i + 1 < argc) { model_path = argv[++i]; continue; }
        if (std::strcmp(arg, "-p") == 0 && i + 1 < argc) { prompt     = argv[++i]; continue; }
        if (std::strcmp(arg, "-b") == 0 && i + 1 < argc) { backend_id = argv[++i]; continue; }
        if (std::strcmp(arg, "-n") == 0 && i + 1 < argc) { max_new_tokens = std::atoi(argv[++i]); continue; }
        if (std::strcmp(arg, "-t") == 0 && i + 1 < argc) { temperature    = std::atof(argv[++i]); continue; }
        if (std::strcmp(arg, "--top-k") == 0 && i + 1 < argc) { top_k  = std::atoi(argv[++i]); continue; }
        if (std::strcmp(arg, "--top-p") == 0 && i + 1 < argc) { top_p  = std::atof(argv[++i]); continue; }
        std::fprintf(stderr, "Unknown argument: %s\n", arg);
        print_usage(argv[0]);
        return 1;
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "Error: -m <model.gguf> is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (prompt.empty()) {
        std::fprintf(stderr, "Error: -p <prompt> is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // ---- Engine setup ----
    axonforge::EngineConfig eng_cfg;
    eng_cfg.backend = backend_id;

    axonforge::SamplerConfig smp_cfg;
    smp_cfg.temperature = temperature;
    smp_cfg.top_k       = top_k;
    smp_cfg.top_p       = top_p;

    std::printf("[AxonForge] Loading model: %s\n", model_path.c_str());
    std::printf("[AxonForge] Backend: %s\n", backend_id.c_str());

    try {
        auto engine  = axonforge::Engine::from_gguf(model_path, eng_cfg);
        auto session = engine.new_session();
        auto kv      = session.new_kv_state();

        std::printf("[AxonForge] Prompt: %s\n\n", prompt.c_str());

        auto prompt_ids = engine.encode(prompt, /*add_bos=*/true);
        auto output_ids = session.generate(prompt_ids, kv, smp_cfg,
                                           max_new_tokens,
                                           engine.eos_id());

        std::printf("%s\n", engine.decode(output_ids).c_str());

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[AxonForge] Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
