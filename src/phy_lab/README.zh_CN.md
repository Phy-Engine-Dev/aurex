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

默认（更安全）：只要评论中包含配置的提及标签（默认 `@aurex`）就会触发回复，并以自然语言进行对话与工具调用。

可选（调试/高级用户）：启用 `!command` 指令模式，需要在配置里设置：

- `agent.commands_enabled=true`

如果你希望在 `User:*` 留言板场景下（纯聊天）不需要 `@aurex` 也能触发回复，可以设置：

- `agent.user_targets_require_mention=false`

补充：即使开启了 `require_mention=true`，当对方是“回复你/回复 aurex”（评论的 `ReplyID` 指向 aurex 的用户 ID）时，Agent 也会触发回复（用于覆盖某些客户端 @ 提及不显示在纯文本里的情况）。

如果你希望启用外网搜索（Google），可以设置：

- `agent.web_search_enabled=true`
- `agent.web_search_proxy=http://127.0.0.1:7897`（选配）
- `agent.auto_web_search=true`（让模型自动判断何时需要联网搜索）

如果你希望让 Agent 在自然语言下自动选择工具（例如生成电路/搜索/总结），可以设置：

- `agent.auto_tool_routing=true`

如果你希望在用户明确表达“发布/分享/投稿”意图时自动发布生成的实验，可以设置：

- `agent.enable_publish=true`
- `agent.auto_publish=true`
- `agent.circuit_max_attempts=3`（编译失败最多重试次数）
- `agent.publish_max_elements=5000`（超过此元件数量将拒绝发布；物实社区发布上限）

如果你担心短时间内请求过多导致负载过高，可以启用“过载保护”。当 10 分钟内请求数量超过阈值时，新请求会直接回复英文忙碌提示：

- `agent.overload_protection_enabled=true`
- `agent.overload_window_sec=600`
- `agent.overload_max_requests=40`
- `agent.overload_message_en="Too many requests at the moment, please try again later."`

自然语言示例：

```text
@aurex 介绍一下这个实验
@aurex summarize 这是一段很长的文章……
@aurex search 逻辑电路
@aurex simulate V=5 R1=100ohm R2=200ohm
@aurex 生成电路 帮我实现一个围棋 9x9 的输入输出接口……
```

当启用 `agent.commands_enabled=true` 时，可用指令：

- `!help`：显示帮助
- `!chat <内容>`：对话
- `!summarize <文本>`：总结（请直接粘贴文本；当前不会自动抓取网页）
- `!summarize`：当你在实验/讨论目标下使用时，会基于当前内容的上下文信息生成总结
- `!search <关键词>`：搜索（仅对“近期作品”做 best-effort 扫描匹配，受 API 限制）
- `!circuit <需求>`：生成电路（Verilog→`.sav`），并在允许发布时自动发布
- `!simulate <参数>`：直流仿真演示（目前支持 VDC + 两个电阻串联）

指令示例：

```text
@aurex !chat 解释一下 RC 低通的截止频率
@aurex !summarize 这是一段很长的文章……
@aurex !search 逻辑电路
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

## 许可证

见 `src/phy_lab/LICENSE`（私有/专有许可证）。
