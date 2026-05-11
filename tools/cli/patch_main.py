#!/usr/bin/env python3
"""Patch tools/cli/main.cpp: add --chat/--system and format_chat() support.
Uses [LT]/[GT]/[SL]/[PIPE] placeholders so this source file has zero literal
angle-bracket characters (works around XML-parameter truncation in the tool).
"""
import os, sys

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.cpp')
with open(TARGET) as f:
    src = f.read()

def un(s):
    return (s
            .replace('[LT]', chr(60))
            .replace('[GT]', chr(62))
            .replace('[SL]', chr(47))
            .replace('[PIPE]', chr(124)))

# Unicode box-drawing helpers
H = '\u2500'   # single horizontal line char used in comments

# ── 1.  Insert chat-template helpers before "Shared generation core" ──────────

MARKER = '// ' + H + H + ' Shared generation core'
assert MARKER in src, "Cannot find shared-core marker"

CHAT = un(
    '// ' + H + H + ' Chat template helpers '
    + H * 78 + '\n\n'
    'enum class ChatFmt { Raw, Zephyr, LLaMA2, ChatML };\n'
    '\n'
    'static ChatFmt detect_chat_fmt(const std::string& tmpl) noexcept {\n'
    '    if (tmpl.empty())                                      return ChatFmt::Raw;\n'
    '    if (tmpl.find("[LT][PIPE]im_start[PIPE][GT]")  != std::string::npos) return ChatFmt::ChatML;\n'
    '    if (tmpl.find("[LT][PIPE]system[PIPE][GT]")    != std::string::npos) return ChatFmt::Zephyr;\n'
    '    if (tmpl.find("[INST]")                        != std::string::npos) return ChatFmt::LLaMA2;\n'
    '    return ChatFmt::Raw;\n'
    '}\n'
    '\n'
    'struct ChatMsg { std::string role; std::string text; };\n'
    '\n'
    'static std::string format_chat(ChatFmt fmt,\n'
    '                                const std::string& system_msg,\n'
    '                                const std::vector[LT]ChatMsg[GT]& history,\n'
    '                                const std::string& user_msg)\n'
    '{\n'
    '    std::string out;\n'
    '    switch (fmt) {\n'
    '\n'
    '    case ChatFmt::Zephyr: {\n'
    '        if (!system_msg.empty()) {\n'
    '            out += "[LT][PIPE]system[PIPE][GT]\\n"; out += system_msg; out += "[LT][SL]s[GT]\\n";\n'
    '        }\n'
    '        for (const auto& msg : history) {\n'
    '            if (msg.role == "user") {\n'
    '                out += "[LT][PIPE]user[PIPE][GT]\\n"; out += msg.text; out += "[LT][SL]s[GT]\\n";\n'
    '            } else {\n'
    '                out += "[LT][PIPE]assistant[PIPE][GT]\\n"; out += msg.text; out += "[LT][SL]s[GT]\\n";\n'
    '            }\n'
    '        }\n'
    '        out += "[LT][PIPE]user[PIPE][GT]\\n"; out += user_msg; out += "[LT][SL]s[GT]\\n";\n'
    '        out += "[LT][PIPE]assistant[PIPE][GT]\\n";\n'
    '        break;\n'
    '    }\n'
    '\n'
    '    case ChatFmt::LLaMA2: {\n'
    '        bool first = true;\n'
    '        for (size_t i = 0; i [LT] history.size(); i += 2) {\n'
    '            const std::string& u = history[i].text;\n'
    '            const std::string  a = (i + 1 [LT] history.size()) ? history[i+1].text : "";\n'
    '            if (first && !system_msg.empty()) {\n'
    '                out += "[INST] [LT][LT]SYS[GT][GT]\\n"; out += system_msg;\n'
    '                out += "\\n[LT][LT][SL]SYS[GT][GT]\\n\\n";\n'
    '                out += u; out += " [/INST] "; first = false;\n'
    '            } else {\n'
    '                out += "[INST] "; out += u; out += " [/INST] ";\n'
    '            }\n'
    '            if (!a.empty()) { out += a; out += " [LT][SL]s[GT][LT]s[GT]"; }\n'
    '        }\n'
    '        if (first && !system_msg.empty()) {\n'
    '            out += "[INST] [LT][LT]SYS[GT][GT]\\n"; out += system_msg;\n'
    '            out += "\\n[LT][LT][SL]SYS[GT][GT]\\n\\n";\n'
    '            out += user_msg; out += " [/INST]";\n'
    '        } else {\n'
    '            out += "[INST] "; out += user_msg; out += " [/INST]";\n'
    '        }\n'
    '        break;\n'
    '    }\n'
    '\n'
    '    case ChatFmt::ChatML: {\n'
    '        if (!system_msg.empty()) {\n'
    '            out += "[LT][PIPE]im_start[PIPE][GT]system\\n"; out += system_msg;\n'
    '            out += "[LT][PIPE]im_end[PIPE][GT]\\n";\n'
    '        }\n'
    '        for (const auto& msg : history) {\n'
    '            out += "[LT][PIPE]im_start[PIPE][GT]"; out += msg.role; out += "\\n";\n'
    '            out += msg.text; out += "[LT][PIPE]im_end[PIPE][GT]\\n";\n'
    '        }\n'
    '        out += "[LT][PIPE]im_start[PIPE][GT]user\\n"; out += user_msg;\n'
    '        out += "[LT][PIPE]im_end[PIPE][GT]\\n";\n'
    '        out += "[LT][PIPE]im_start[PIPE][GT]assistant\\n";\n'
    '        break;\n'
    '    }\n'
    '\n'
    '    default:\n'
    '        out = user_msg;\n'
    '        break;\n'
    '    }\n'
    '    return out;\n'
    '}\n\n'
)

src = src.replace(MARKER, CHAT + MARKER, 1)
assert 'enum class ChatFmt' in src, "Chat helpers not inserted"

# ── 2.  run_interactive signature ─────────────────────────────────────────────

OLD_SIG = (
    'static void run_interactive(const axonforge::Engine& engine,\n'
    '                             const std::string& prefix,\n'
    '                             int max_new_tokens, float temperature, int top_k,\n'
    '                             bool verbose) {'
)
assert OLD_SIG in src, "Cannot find run_interactive signature"

NEW_SIG = (
    'static void run_interactive(const axonforge::Engine& engine,\n'
    '                             const std::string& prefix,\n'
    '                             int max_new_tokens, float temperature, int top_k,\n'
    '                             bool verbose,\n'
    '                             ChatFmt chat_fmt,\n'
    '                             const std::string& system_msg) {'
)
src = src.replace(OLD_SIG, NEW_SIG, 1)

# ── 3.  Conversation state ────────────────────────────────────────────────────

OLD_ST = un(
    '    std::vector[LT]int32_t[GT] history_ids;\n'
    '    const int max_context  = mc.max_seq_len [GT] 0 ? std::min(mc.max_seq_len, 4096) : 4096;\n'
    '    const int history_cap  = max_context - max_new_tokens - 16;  // leave room for decode'
)
assert OLD_ST in src, "Cannot find conversation-state section"

NEW_ST = un(
    '    const int max_context = mc.max_seq_len [GT] 0 ? std::min(mc.max_seq_len, 4096) : 4096;\n'
    '    const int history_cap = max_context - max_new_tokens - 16;\n'
    '    std::vector[LT]ChatMsg[GT]  chat_history;   // chat mode: (role, text) pairs\n'
    '    std::vector[LT]int32_t[GT]  history_ids;    // raw mode: accumulated token IDs'
)
src = src.replace(OLD_ST, NEW_ST, 1)
assert 'chat_history' in src, "Conversation state not updated"

# ── 4.  /new handler ──────────────────────────────────────────────────────────

OLD_NEW = (
    '        if (line == "/new") {\n'
    '            history_ids.clear();\n'
    '            if (colours) std::printf("%s", COL_STATS);\n'
    '            std::printf("[Conversation history cleared. Starting fresh.]\\n\\n");\n'
    '            if (colours) std::printf("%s", COL_RESET);\n'
    '            continue;\n'
    '        }'
)
assert OLD_NEW in src, "Cannot find /new handler"

NEW_NEW = (
    '        if (line == "/new") {\n'
    '            history_ids.clear();\n'
    '            chat_history.clear();\n'
    '            if (colours) std::printf("%s", COL_STATS);\n'
    '            std::printf("[Conversation history cleared. Starting fresh.]\\n\\n");\n'
    '            if (colours) std::printf("%s", COL_RESET);\n'
    '            continue;\n'
    '        }'
)
src = src.replace(OLD_NEW, NEW_NEW, 1)

# ── 5.  Build-prompt + generate + update-history block ───────────────────────

# Locate section boundaries robustly using index()
BUILD_START = '        // ' + H + H + ' Build prompt with history'
AFTER_HIST  = (
    '        history_ids = std::move(prompt_ids);\n'
    '        history_ids.insert(history_ids.end(), gen_ids.begin(), gen_ids.end());'
)

i_build = src.index(BUILD_START)
i_after = src.index(AFTER_HIST, i_build) + len(AFTER_HIST)

NEW_BLK = un(
    '        // ' + H + H + ' Build prompt ' + H * 81 + '\n'
    '        const std::string user_text = prefix.empty() ? line : prefix + line;\n'
    '        std::vector[LT]int32_t[GT] prompt_ids;\n'
    '        if (chat_fmt != ChatFmt::Raw) {\n'
    '            const std::string full_prompt = format_chat(chat_fmt, system_msg, chat_history, user_text);\n'
    '            prompt_ids = engine.encode(full_prompt, /*add_bos=*/true);\n'
    '        } else {\n'
    '            auto user_ids = engine.encode(user_text, /*add_bos=*/false);\n'
    '            if ((int)(history_ids.size() + user_ids.size()) [GT] history_cap) {\n'
    '                const int excess = (int)(history_ids.size() + user_ids.size()) - history_cap;\n'
    '                if (excess [LT] (int)history_ids.size())\n'
    '                    history_ids.erase(history_ids.begin(), history_ids.begin() + excess);\n'
    '                else\n'
    '                    history_ids.clear();\n'
    '                if (verbose) std::fprintf(stderr, "[context window trimmed]\\n");\n'
    '            }\n'
    '            prompt_ids.reserve(history_ids.size() + user_ids.size());\n'
    '            prompt_ids.insert(prompt_ids.end(), history_ids.begin(), history_ids.end());\n'
    '            prompt_ids.insert(prompt_ids.end(), user_ids.begin(), user_ids.end());\n'
    '        }\n'
    '\n'
    '        // ' + H + H + ' Generate ' + H * 85 + '\n'
    '        const auto gen_ids = generate_and_print(\n'
    '            engine, prompt_ids, max_new_tokens, temperature, top_k,\n'
    '            /*show_stats=*/true, verbose, colours);\n'
    '\n'
    '        // ' + H + H + ' Update history ' + H * 79 + '\n'
    '        if (chat_fmt != ChatFmt::Raw) {\n'
    '            const std::string asst_text = engine.decode({gen_ids.data(), gen_ids.size()});\n'
    '            chat_history.push_back({"user",      user_text});\n'
    '            chat_history.push_back({"assistant", asst_text});\n'
    '        } else {\n'
    '            history_ids = std::move(prompt_ids);\n'
    '            history_ids.insert(history_ids.end(), gen_ids.begin(), gen_ids.end());\n'
    '        }'
)

src = src[:i_build] + NEW_BLK + src[i_after:]

# ── 6.  main() variable declarations ─────────────────────────────────────────

OLD_VARS = (
    '    bool        interactive    = false;\n'
    '    bool        verbose        = false;'
)
assert OLD_VARS in src, "Cannot find main() bool variables"

NEW_VARS = (
    '    bool        interactive    = false;\n'
    '    bool        verbose        = false;\n'
    '    bool        chat_mode      = false;\n'
    '    std::string system_msg     = "You are a helpful assistant.";'
)
src = src.replace(OLD_VARS, NEW_VARS, 1)

# ── 7.  main() flag parsing ───────────────────────────────────────────────────

OLD_FLAG = (
    '        if ((std::strcmp(arg, "-V") == 0 || std::strcmp(arg, "--verbose") == 0))\n'
    '            { verbose = true; continue; }'
)
assert OLD_FLAG in src, "Cannot find verbose flag"

NEW_FLAG = un(
    '        if ((std::strcmp(arg, "-V") == 0 || std::strcmp(arg, "--verbose") == 0))\n'
    '            { verbose = true; continue; }\n'
    '        if (std::strcmp(arg, "--chat") == 0) { chat_mode = true; continue; }\n'
    '        if (std::strcmp(arg, "--system") == 0 && i + 1 [LT] argc)\n'
    '            { system_msg = argv[++i]; continue; }'
)
src = src.replace(OLD_FLAG, NEW_FLAG, 1)

# ── 8.  run_interactive call in main() ────────────────────────────────────────

OLD_CALL = '            run_interactive(engine, prefix, max_new_tokens, temperature, top_k, verbose);'
assert OLD_CALL in src, "Cannot find run_interactive call"

NEW_CALL = (
    '            const ChatFmt fmt = chat_mode\n'
    '                ? detect_chat_fmt(engine.chat_template())\n'
    '                : ChatFmt::Raw;\n'
    '            if (chat_mode && fmt == ChatFmt::Raw)\n'
    '                std::fprintf(stderr,\n'
    '                    "[AxonForge] Warning: --chat used but no chat template found in model; raw mode.\\n");\n'
    '            run_interactive(engine, prefix, max_new_tokens, temperature, top_k,\n'
    '                            verbose, fmt, system_msg);'
)
src = src.replace(OLD_CALL, NEW_CALL, 1)

# ── 9.  Write ─────────────────────────────────────────────────────────────────

with open(TARGET, 'w') as f:
    f.write(src)
print('Patched', TARGET, '(' + str(src.count(chr(10))) + ' lines)')
