# phy_lab（物理实验室 AR 自动回复 Agent）

`phy_lab` 是一个 Python Agent：使用邮箱+密码登录物理实验室 AR 账号，轮询指定评论区（用户主页/实验/讨论），并通过本地 Ollama 模型自动生成回复。它还支持工具型指令：对话、总结、搜索（受 API 限制的“近似搜索”），以及可选的 Verilog→`.sav` 电路生成与发布（依赖 `third-parties/Phy-Engine` 的 `verilog2plsav`）。

## 运行环境

- Python 3.10+
- Ollama 本地服务（默认 `http://127.0.0.1:11434`）
- 物理实验室 AR 邮箱账号
- 可选：CMake（仅当你启用自动编译 Phy-Engine）

## 安装依赖

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r src/phy_lab/requirements.txt
```

## 启动 Ollama

确保 Ollama 在本机运行，并准备好模型（示例使用 `llama3.1`）：

```bash
ollama serve
ollama pull llama3.1
```

## 创建配置

交互式创建（只写入邮箱，不会保存密码；密码运行时输入）：

```bash
python src/phy_lab/agent.py init --config .phy_lab/config.json
```

你也可以直接参考并复制示例配置：

- `src/phy_lab/config.example.json`

如果你使用 `gpt-oss:*` 模型，并希望强化推理质量，可以在 `ollama` 下开启：

- `ollama.gptoss-optimization=true`（会注入 Harmony 风格头部 `reasoning: high`；不会改变工具/输出格式要求）

## 运行（推荐先 dry-run）

单次轮询、不发帖（最安全）：

```bash
python src/phy_lab/agent.py run --config .phy_lab/config.json --once --dry-run
```

持续运行（会发帖，除非你指定 `--dry-run` 或配置里 `agent.dry_run=true`）：

```bash
python src/phy_lab/agent.py run --config .phy_lab/config.json
```

## 评论区交互方式

本项目支持两种模式（`agent.mode`）：

- `agent`（推荐）：把每条触发的评论当作“代理任务”来完成（可多步调用站内查询/仿真/编译/联网搜索等），最终给出结论。
- `traditional`：传统机器人模式（对话/总结/搜索/生成电路/仿真等），不提供代理接口。

可选（调试/高级用户）：启用 `!command` 指令模式，需要在配置里设置：

- `agent.commands_enabled=true`

如果你希望在 `User:*` 留言板场景下（纯聊天）不需要 `@aurex` 也能触发回复，可以设置：

- `agent.user_targets_require_mention=false`

补充：如果你开启了 `agent.trigger_on_reply_to_self=true`，即使 `require_mention=true`，当对方是“回复你/回复 aurex”（评论的 `ReplyID` 指向 aurex 的用户 ID）时，Agent 也会触发回复（用于覆盖某些客户端 @ 提及不显示在纯文本里的情况）。默认建议关闭以避免刷屏。

如果你希望避免在公共评论区被“多人接力回复”刷屏，可以开启一次性回复策略（默认开启）：

- `agent.reply_once=true`：每个（target,user）只回复一次，之后对话会被关闭
- `agent.reply_once_ttl_sec=0`：不启用冷却（推荐；你后续再次 `@aurex` 仍然会回复）
- `agent.reply_once_ttl_sec=30`（示例）：启用冷却 30 秒；在冷却内同一（target,user）会被跳过
- `agent.trigger_on_reply_to_self=false`：关闭“仅通过回复 aurex 就触发”的机制（避免非提问者把你拉进对话）

站内“可读性/浏览”能力（`agent` 模式下会自动组合使用）：

- 列表浏览：`list_plar latest|hot|featured|random`
- 社交关系列表（与 plweb2 Friends 页一致）：`list_plar following|followers|banned|volunteers|editors|retired`
- 打开并读取：`plar_get_user_board`（读取用户留言板）、`plar_open_content_page`（读取实验/讨论 + 最近评论）

如果你希望启用外网搜索（联网检索 + 结合结果回答），可以设置：

- `agent.web_search_enabled=true`
- `agent.web_search_provider=duckduckgo-search|bing|baidu|google|searxng`（建议：`duckduckgo-search`：不需要 API Key；更稳定的方案是自建 `searxng`）
- `agent.web_search_searxng_base_url=http://127.0.0.1:8080`（当 provider=searxng 时使用）
- `agent.web_search_user_agent=`（选配，自定义 UA）
- `agent.web_search_proxy=http://127.0.0.1:7897`（选配）
- `agent.auto_web_search=true`（让模型自动判断何时需要联网搜索）

当你选择 `duckduckgo-search` 时，需要安装依赖：`pip install duckduckgo-search`

补充：物实社区不提供“全站关键词搜索”。因此站内定位/发现通常只能通过：
- 列表型浏览（最新/最热门/精选）+ 人工筛选
- 通过 ID/用户名直接定位（用户 ID、实验/讨论 ID）

登录密码不建议写入配置文件。需要无人值守运行时，请使用环境变量（优先级最高）：
- `PHY_LAB_PASSWORD` 或 `PHYSICSLAB_PASSWORD`

测试账号/本地调试如果想避免交互输入，也可以在配置文件里写入 `account.password`（优先级低于环境变量）。

调试命令（需要登录）：
- `phy_lab-agent apitest --config .phy_lab/config.json --take 5 --user-name 某个昵称 --query 关键词`
- `phy_lab-agent publishsav --config .phy_lab/config.json --sav-path path/to/design.sav --category Discussion --title 标题 --yes`

如果你希望让 Agent 在自然语言下自动选择工具（例如生成电路/搜索/总结），可以设置：

- `agent.auto_tool_routing=true`

如果你希望在用户明确表达“发布/分享/投稿”意图时自动发布生成的实验，可以设置：

- `agent.enable_publish=true`
- `agent.auto_publish=true`
- `agent.circuit_max_attempts=3`（编译失败最多重试次数）
- `agent.publish_max_elements=5000`（超过此元件数量将拒绝发布；物实社区发布上限）

如果你担心短时间内请求过多导致负载过高，可以启用“过载保护”。当 10 分钟内请求数量超过阈值时，新请求会直接回复英文忙碌提示：

- `agent.overload_protection_enabled=true`

如果你希望启用“AI 自动构建电路并仿真”（当内置串联解析/实验 StatusSave 仿真无法满足时，模型会生成一个小电路并调用 Phy-Engine 仿真），可以设置：

- `agent.simulation_ai_enabled=true`
- `agent.simulation_ai_max_components=30`（AI 构建电路的最大元件数）
- `agent.simulation_ai_max_probes=20`（最多输出多少行探针结果）
- `agent.overload_window_sec=600`
- `agent.overload_max_requests=40`
- `agent.overload_message_en="Too many requests at the moment, please try again later."`

自然语言示例：

```text
@aurex 介绍一下这个实验
@aurex summarize 这是一段很长的文章……
@aurex agent 帮我先在站内搜索“运放”，再结合网页资料总结常见用法
@aurex search 逻辑电路
@aurex simulate V=5 R1=100ohm R2=200ohm
@aurex 生成电路 帮我实现一个围棋 9x9 的输入输出接口……
```

当 `agent.mode=traditional` 且启用 `agent.commands_enabled=true` 时，可用指令：

- `!help`：显示帮助
- `!chat <内容>`：对话
- `!summarize <文本>`：总结（请直接粘贴文本；当前不会自动抓取网页）
- `!summarize`：当你在实验/讨论目标下使用时，会基于当前内容的上下文信息生成总结
- `!search <@用户名|uid:<id>|experiment:<id>|discussion:<id>>`：仅支持“定位/查询”（不支持关键词搜索）
- `!circuit <需求>`：生成电路（Verilog→`.sav`），并在允许发布时自动发布
- `!simulate <参数>`：直流仿真演示（目前支持 VDC + 两个电阻串联）

指令示例：

```text
@aurex !chat 解释一下 RC 低通的截止频率
@aurex !summarize 这是一段很长的文章……
@aurex !search @someone
@aurex !search experiment:0123456789abcdef01234567
@aurex !simulate V=5 R1=100ohm R2=200ohm
@aurex !circuit 帮我实现一个围棋 9x9 的输入输出接口……
```

## 启用 Verilog→`.sav`（Phy-Engine / verilog2plsav）

两种方式：

1) 你已经编译好了 `verilog2plsav`：
   - 在配置里设置 `phy_engine.verilog2plsav_path` 指向该可执行文件路径

2) 让 Agent 自动编译：
   - 配置 `phy_engine.auto_build=true`
   - 确保 `cmake` 在 PATH 中可用
   - 编译输出将放到配置目录下的缓存目录（默认 `.phy_lab/cache/phy-engine-build`）

注意：电路“发布”功能默认关闭。需要显式设置：

- `agent.enable_publish=true`

强烈建议先用 `--dry-run` 验证行为。

## 启用本地仿真（Phy-Engine / phyengine）

直流仿真演示依赖 `third-parties/Phy-Engine` 的共享库目标 `phyengine`。

两种方式：

1) 你已经编译好了共享库（例如 `libphyengine.so`）：
   - 在配置里设置 `phy_engine.phyengine_lib_path` 指向该文件路径

2) 让 Agent 自动编译：
   - 配置 `phy_engine.auto_build=true`
   - 确保 `cmake` 在 PATH 中可用

## 缓存与清理

Agent 会把所有临时/缓存文件放到同一个缓存目录下（由 `storage.cache_dir` 决定，默认相对配置目录的 `cache/`）。同时它会把 `PHYSICSLAB_HOME_PATH` 指向缓存子目录，避免 `physicsLab` 把 `.sav` 写到不受控位置。

清理缓存：

```bash
python src/phy_lab/agent.py cleanup --config .phy_lab/config.json
```

## 上下文增强（实验/讨论）

当 Agent 监控的是 `Experiment` 或 `Discussion` 目标时，它会尽力拉取并注入上下文：

- 作品的标题/简介（summary）
- `get_experiment` 返回的数据（并提取一个小型 `.sav` 摘要，例如元件/导线数量等）

上下文会缓存到 `storage.cache_dir/plar_cache/` 下，用于减少重复请求。

## 日志

日志会写入 `storage.cache_dir/logs/phy_lab.log`。如果你需要在控制台看到更详细的信息：

```bash
python src/phy_lab/agent.py run --config .phy_lab/config.json --log-level DEBUG
```

如果你需要把“评论原文”也写入 debug 日志，请在配置中设置 `agent.log_include_comment_content=true`（注意隐私风险）。
如果你需要更详细的 Agent 工具调用轨迹（安全截断，不记录模型思维链），可以设置：
- `agent.debug_log_llm_io=true`
- `agent.debug_llm_max_chars=800`

## 联网搜索排查

如果你发现联网搜索（Google/Baidu/DuckDuckGo 等）返回空结果或被验证码拦截，建议：

- 尝试 `agent.web_search_provider=duckduckgo-search` 或 `agent.web_search_provider=bing`
- 或者自建 SearXNG：`agent.web_search_provider=searxng` 并设置 `agent.web_search_searxng_base_url=http://127.0.0.1:8080`

你也可以直接跑一个无需登录的联网测试命令，快速判断是“网络不可达/代理问题/验证码拦截/解析失败”：

```bash
python src/phy_lab/agent.py webtest --config .phy_lab/config.json --all --query "你的问题"
```

## 仍然不回复时如何排查

如果你确认目标评论区正确，但 Agent 仍然没有触发回复，建议先用诊断命令检查：

```bash
python src/phy_lab/agent.py diagnose --config .phy_lab/config.json --take 10
```

它会拉取每个 target 的最近评论，并输出 `mention/prefix/triggered` 判定结果，帮助定位是“没拉到评论”还是“没识别到 @aurex”。

补充：如果你希望覆盖“被 @ 提及但不在你配置的 targets 里”的情况，Agent 会通过通知接口自动发现目标（`Messages/GetMessages`）。你可以在 `diagnose` 输出里看到 `Messages:*` 段落，里面会显示每条通知能否解析出 `target=Experiment:.../Discussion:...` 等信息。

如果你更新了 Agent 代码，但之前的通知已经被记录为“已处理”，可以重置本地状态让它重新处理最近通知：

```bash
python src/phy_lab/agent.py reset-state --config .phy_lab/config.json --scope notifications
```

## 电路编译失败如何排查

当你让 aurex 生成电路并编译 `.sav` 时，如果 `verilog2plsav` 编译失败，Agent 会把“失败工件”保存到缓存目录中，方便你离线复现与排查：

- `storage.cache_dir/artifacts/<artifact_id>/design.v`
- `storage.cache_dir/artifacts/<artifact_id>/compile_error.txt`

## 许可证

见 `src/phy_lab/LICENSE`（私有/专有许可证）。
