# phy_lab (Industrial Auto-Reply Agent for Physics Lab AR)

`phy_lab` is a Python agent that logs into a Physics Lab AR account, monitors comment boards, and replies using a local Ollama model. It can also run tool workflows (summarization, basic experiment search, and optional Verilog→`.sav` generation via Phy-Engine).

This repository vendors `physicsLab` under `third-parties/physicsLab` and a CMake-buildable `Phy-Engine` under `third-parties/Phy-Engine`. `phy_lab` treats both as dependencies and keeps its own runtime artifacts in a controlled cache directory.

Chinese documentation is available in `src/phy_lab/README.zh_CN.md`.

## Goals

- Safe-by-default: no password persistence, explicit opt-in for publish workflows
- Deterministic storage: all cache/temp files go to one directory and are removable
- Operational controls: `--dry-run`, `--once`, rate limiting, timeouts
- Mention coverage: uses the official Notifications API to discover where you were mentioned

## Requirements

- Python 3.10+
- Ollama running locally (default: `http://127.0.0.1:11434`)
- A Physics Lab AR email-based account
- Optional: CMake toolchain (only if you enable Phy-Engine auto-build)

## Install Dependencies

This agent uses the vendored `physicsLab` client (which depends on `requests`) and calls Ollama via HTTP.

Recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r src/phy_lab/requirements.txt
```

## Start Ollama

Ensure Ollama is running and the configured model exists:

```bash
ollama serve
ollama pull llama3.1
```

## Quick Start

Create config:

```bash
python src/phy_lab/agent.py init --config .phy_lab/config.json
```

Run (prompts for password; does not store it):

```bash
python src/phy_lab/agent.py run --config .phy_lab/config.json
```

Safe trial (no posting, single cycle):

```bash
python src/phy_lab/agent.py run --config .phy_lab/config.json --once --dry-run
```

## Configuration

Example configuration: `src/phy_lab/config.example.json`.

Key settings:

- `account.email`: login email (password is prompted at runtime)
- `ollama.base_url`, `ollama.model`: Ollama endpoint/model
- `agent.require_mention`: when `true`, only replies to comments that contain `agent.mention_tag` (and optionally commands if enabled)
- `agent.user_targets_require_mention`: override mention requirement for `User:*` targets (default: `true`; set `false` for chat-like message boards)
- Trigger note: even with `require_mention=true`, the agent will reply when a comment is a direct reply to the agent account (i.e., comment `ReplyID` matches the agent `UserID`).
- `agent.commands_enabled`: enable the `!command` interface (default: `false` for safety)
- `agent.notifications_enabled`: poll the Notifications API and auto-discover targets (default: `true`)
- `agent.notification_category_ids`: which notification categories to poll (default: `[0, 3]`)
- `agent.web_search_enabled`: enable Google web search (default: `false`)
- `agent.web_search_proxy`: optional HTTP(S) proxy (example: `http://127.0.0.1:7897`)
- `agent.auto_web_search`: let the LLM decide when to use web search (default: `true`, only effective when `web_search_enabled=true`)
- `agent.web_search_fallback_to_ddg`: fallback to DuckDuckGo when Google is blocked/captcha (default: `true`)
- `agent.auto_tool_routing`: let the LLM route natural language requests to tools (default: `true`)
- `agent.enable_publish`: allow publishing generated experiments (default: `false`)
- `agent.auto_publish`: allow the LLM to publish when user explicitly requests it (default: `false`; requires `enable_publish=true`)
- `agent.circuit_max_attempts`: max compile retries for circuit generation (default: `3`)
- `agent.publish_max_elements`: refuse publishing if generated `.sav` has more than this many elements (default: `5000`)
- `agent.overload_protection_enabled`: enable overload protection (default: `true`)
- `agent.overload_window_sec`: time window for request counting (default: `600`)
- `agent.overload_max_requests`: max requests per window before replying busy (default: `40`)
- `agent.overload_message_en`: busy reply message (English) sent when overloaded
- `agent.targets`: additional boards to monitor (`User`, `Experiment`, `Discussion`)
- `storage.cache_dir`: the single cache root (relative to the config directory unless absolute)
- `phy_engine.*`: optional `verilog2plsav` / `phyengine` configuration for circuit generation and local simulation
- `agent.bootstrap_lookback_sec`: when state is empty, only process the last N seconds of history (default: `600`)

## Comment Interface

Default mode is natural language only: mention the agent and ask directly.

Optional command mode exists for debugging and power users (`agent.commands_enabled=true`).

Supported interactions (natural language):

- `@aurex <question>` — chat with full page context (when on Experiment/Discussion)
- `@aurex summarize <text>` — summarize
- `@aurex search <query>` — search recent experiments (best effort)
- `@aurex simulate V=5 R1=100ohm R2=200ohm` — DC simulation demo (requires `phy_engine.auto_build=true` or `phy_engine.phyengine_lib_path`)
- `@aurex generate circuit <spec>` / `@aurex circuit <spec>` — generate Verilog + `.sav` (publishing requires explicit enablement in config)
- `@aurex google <query>` — Google web search (requires `agent.web_search_enabled=true`)

Examples:

- `@aurex Explain RC cutoff frequency`
- `@aurex Summarize this experiment`
- `@aurex search logic circuit`
- `@aurex generate circuit Build a 9x9 Go board input/output interface`

If you enable command mode (`agent.commands_enabled=true`), `!help`, `!chat`, `!summarize`, `!search`, `!circuit`, `!simulate` are also available.

## Circuit Generation (Verilog → .sav)

`!circuit` uses a local LLM to generate Verilog, then calls `verilog2plsav` from `third-parties/Phy-Engine` to produce a Physics Lab `.sav`. Publishing is disabled by default.

Enable circuit generation:

- Provide a prebuilt `verilog2plsav` binary:
  - Set `phy_engine.verilog2plsav_path` to the executable path
- Or enable auto-build:
  - Set `phy_engine.auto_build=true` and ensure `cmake` is available
  - Build output is stored under `storage.cache_dir` and can be removed by `cleanup`

Recommended defaults for high-quality layout/optimization:

- `phy_engine.verilog2plsav_args=["-O4","--layout","hier"]`
- `phy_engine.prebuild_on_start=true` (optional; compiles on startup to reduce latency)

## Local Simulation (Phy-Engine)

The included DC simulation demo uses the `phyengine` shared library (`third-parties/Phy-Engine`, target `phyengine`).

Enable simulation:

- Provide a prebuilt shared library:
  - Set `phy_engine.phyengine_lib_path` (e.g., `.../libphyengine.so`)
- Or enable auto-build:
  - Set `phy_engine.auto_build=true` and ensure `cmake` is available

Enable publishing (explicit opt-in):

- Set `agent.enable_publish=true`
- If you want natural-language requests to auto-publish, also set `agent.auto_publish=true`
- Validate first with `--dry-run`

## Cache and Cleanup

`phy_lab` uses a single cache root (default: `.phy_lab/cache`). It also sets `PHYSICSLAB_HOME_PATH` to a subdirectory of the cache to prevent `physicsLab` from writing to uncontrolled locations.

Cleanup:

```bash
python src/phy_lab/agent.py cleanup --config .phy_lab/config.json
```

## Context Enrichment (Experiment/Discussion)

When the agent is monitoring an `Experiment` or `Discussion` target, it will best-effort fetch:

- The content summary (title/description)
- The content payload via `get_experiment` and a small `.sav`-like summary (element/wire counts if available)

This context is injected into the LLM prompt for `@mention`/command-triggered replies and cached under `storage.cache_dir/plar_cache/`.

## Troubleshooting

- `Missing dependency: requests`: run `python -m pip install -r src/phy_lab/requirements.txt`
- `Failed to reach Ollama`: confirm `ollama serve` is running and `ollama.base_url` is correct
- `Login failed`: verify the account email/password and ensure the API is reachable
- Circuit build failures: set `phy_engine.auto_build=true` (or point to a working `verilog2plsav`) and check CMake output in the cache directory
- Agent does not reply: run `python src/phy_lab/agent.py diagnose --config .phy_lab/config.json --take 10` to verify the agent can fetch the target comments and detect `@mentions`
- Agent still does not reply after a mention: run `python src/phy_lab/agent.py diagnose --config .phy_lab/config.json` and check the `Messages:*` sections to confirm notifications are being fetched and targets are being discovered
- After updating the agent, you may want to reset local state to re-process recent notifications: `python src/phy_lab/agent.py reset-state --config .phy_lab/config.json --scope notifications`

## Logging

Logs are written to `storage.cache_dir/logs/phy_lab.log`. For more details on the console, run:

```bash
python src/phy_lab/agent.py run --config .phy_lab/config.json --log-level DEBUG
```

To include incoming comment text in debug logs, set `agent.log_include_comment_content=true` in the config.

## License

See `src/phy_lab/LICENSE`.
