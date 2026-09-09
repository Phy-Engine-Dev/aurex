# Aurex 3

面向物理实验室社区的中文电学实验助手。Aurex 通过独立的 `aurex-vllm` 容器调用 Qwen3.8-27B-AWQ-MTP，支持文本、图片、社区检索、模拟/数字/混合电路分析、Verilog 工作区和 PLSAV 导出。开发用途的 `vllm` 容器与 OpenCode 不会被脚本删除。

> 安全提示：仓库绝不保存社区账号密码、API Token、管理员访问码、模型服务密钥或 `.aurex/` 数据库。下面的“账户配置”仅说明私有文件字段和操作方式；请勿把真实凭据写入 Issue、日志或提交。

## 快速部署（2 × V100 16 GiB）

运行主机需要 Linux、Podman、两张编号为 GPU 1/2 的 V100 16 GiB、NVIDIA 容器运行时、Python 3.10+、CMake 与能编译 C++20 的 Clang 或 GCC。用于渲染电路图的 Cairo 系统库也必须可用。以下命令在仓库根目录执行：

```bash
git clone -b aurex3 https://github.com/Phy-Engine-Dev/aurex.git aurex3
cd aurex3
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
# 构建多模态 Qwen 服务；它使用 GPU 1、2，不能与开发 vllm 容器同时运行。
bash scripts/aurex-vllm.sh start
bash scripts/aurex-vllm.sh status

# 生成仅本机可读的配置与管理员访问码，再启动 Web。
.venv/bin/python scripts/configure-aurex3.py
bash scripts/aurex-web.sh start
bash scripts/aurex-web.sh status
```

当前默认服务名是 `qwen38-27b`，API 地址为 `http://127.0.0.1:8000/v1`。容器使用 TP=2、AWQ、V100 专用注意力后端、视觉编码器、90112 token 上下文、2048 token 批处理预算、前缀缓存和仅解码 CUDA Graph；默认关闭思考，由 Aurex 仅在首轮完整上下文、发布审核及最终审核请求中按需开启。

模型 API 可在本机直接使用：

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/models
curl --noproxy '*' http://127.0.0.1:8000/health
```

若已有兼容 OpenAI 的模型服务，不必启动容器；将私有 `.config/aurex3.json` 的 `llm.base_url`、`llm.model` 与实际上下文长度改为该服务的值，然后执行 Web 启动命令即可。

## 账户、角色与 Web

`scripts/configure-aurex3.py` 从 `aurex3.config.example.json` 创建权限为 `0600` 的 `.config/aurex3.json`，并在 `.config/web-token` 创建随机管理员访问码。社区服务账户只填写在私有配置的以下字段中：

```json
{
  "account": {
    "email": "你的社区邮箱",
    "password": "你的社区密码"
  }
}
```

网页默认监听 `0.0.0.0:4097`，提供两种真实的服务端角色：

- 用户模式输入 24 位物理实验室用户 ID，读取并显示该用户的公开昵称、简介、等级和统计。用户 ID 是公开资料标签，不是社区账号认证，也不会授予发布、回复或冒充该用户的权限；会话所有权仍由随机 HttpOnly 浏览器凭据隔离。
- 管理员模式使用 `.config/web-token` 中的访问码，可查看全局会话和完整 FIFO 队列、创建管理员任务并管理任意任务。访问码只能在部署机器上读取，不能公开。

普通用户只能读取和取消自己的任务；全局队列只显示他人的匿名占位、状态与排队位置。历史社区/CLI/管理员记录只对管理员可见。桌面端的会话、当前对话、全局队列为三个同级区域，任务栏可拖动或用方向键调宽并保存宽度；手机端通过“会话/队列”按钮打开全屏面板。

任务调度并行度由 `tracking.max_parallel_tasks` 控制，必须是 `1..64` 的整数，`0`、布尔值及字符串都会拒绝加载。本机的 TP2 单序列 vLLM 配置使用 `1`；只有模型后端、GPU 与社区 API 确实支持并发时才调高。调高后仅不同会话可以并行，同一会话内的任务仍按顺序执行；SQLite 原子领取、唯一服务锁以及发布/回复的一次性回执约束保持不变。

```bash
bash scripts/aurex-web.sh start          # 启动网页、队列与持续社区轮询
bash scripts/aurex-web.sh status         # 健康检查
bash scripts/aurex-web.sh stop           # 停止服务，不删数据库、缓存或模型
```

默认 `agent.dry_run=true`，即使登录成功也不会发送社区评论或发布实验。生产外发前必须在私有配置中明确关闭 dry-run，并理解发布审核与至多一次外发规则。

## 编译 Phy-Engine

Phy-Engine 源码直接内置在 `third-parties/Phy-Engine`，不再使用 Git 子模块，也不需要单独初始化。首次构建或引擎代码更新后执行：

```bash
cmake -S third-parties/Phy-Engine/src -B .aurex/cache/phy-engine-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++-21 \
  -DCMAKE_C_COMPILER=clang-21 \
  -DPHY_ENGINE_USE_LEVELDB=OFF
cmake --build .aurex/cache/phy-engine-build \
  --target phyengine verilog2plsav circuit_view --parallel 2
```

如系统没有 `clang++-21`，可改为已安装的 C++20 编译器。Aurex 在首次调用电路工具时也会尝试自动构建；手动构建更容易定位 CMake、编译器或 Cairo 依赖问题。

## 依赖边界

顶层 `third-parties/` 有两项：

| 项目 | 数量/来源 | 用途 | 是否需要联网下载 |
|---|---:|---|---|
| `Phy-Engine` | 1 份内置源码，基于上游 `aurex` 分支 | 求解、HDL 转 PLSAV、SVG/PNG 电路图 | 否 |
| `physicsLab` | 1 份内置的 MIT 许可源码，固定 2.0.6 | 官方社区登录、读取、评论与发布 API | 否 |

Phy-Engine 自带 4 份 C++ 源码依赖：Eigen、Abseil、fast_io、LevelDB；它们不是嵌套子模块。默认构建关闭 LevelDB。Aurex 将 PhysicsLab 官方 SDK 和官方发布传输代码放在仓库内，`requirements.txt` 不再依赖 PyPI 的 `physicsLab` 包；Python 仍需要 `requests`、`httpx`、Pillow 等通用运行时依赖。

## 定期更新与提交

日常更新前先停止 Web，保留 `.aurex/`，再更新 `aurex3` 分支、构建并运行最小验证。内置的 Phy-Engine 会随 Aurex 一起更新：

```bash
bash scripts/aurex-web.sh stop
git switch aurex3
git pull --ff-only origin aurex3

cmake --build .aurex/cache/phy-engine-build --target phyengine verilog2plsav circuit_view --parallel 2
PYTHONPATH=src .venv/bin/python -m unittest tests.test_circuit_tools tests.test_physicslab_pe_coverage -v
bash scripts/aurex-web.sh start
```

同步 Phy-Engine 上游时，应在临时目录克隆 `Phy-Engine-Dev/Phy-Engine` 的 `aurex` 分支，审阅差异后再把源码更新到 `third-parties/Phy-Engine`，随后在 Aurex 仓库中统一构建、测试和提交。不要复制临时仓库的 `.git`，也不要对有本地改动的部署目录执行 `reset --hard`。向官方目标仓库提交时只推送 `aurex3`，不使用 `codex/*` 分支：

```bash
git add third-parties/Phy-Engine
git commit -m "同步内置 Phy-Engine"
git push origin aurex3
```

## 报告问题

请在 [Aurex Issues](https://github.com/Phy-Engine-Dev/aurex/issues) 报告 Web、Agent、社区工作流和导入问题；在 [Phy-Engine Issues](https://github.com/Phy-Engine-Dev/Phy-Engine/issues) 报告求解器、元件模型、Verilog/PLSAV 转换或渲染问题。Issue 应包含：复现命令、预期与实际结果、Aurex/Phy-Engine 提交号、匿名化后的 `.sav` 或最小电路、`circuit_analyze` 输出及相关日志。不得提交账号密码、访问码、API Token、私有社区内容或完整思考记录。

### 操作员隔离回放

`scripts/aurex-operator-replay.py` 默认 `prepare` 只读既有任务的来源、长度和SHA-256，不输出原文或思考正文。`status --task-id ID --observe` 只读SQLite状态、模型token/上下文指标、工具名/耗时及外发ledger；观察到期不取消、不重新入队。两者以SQLite `mode=ro` / `query_only` 打开库，不触发迁移或改旧记录。

只有明确执行 `enqueue --case cpu --execute` 才复制指定旧task的原始prompt（含CONTEXT_JSON）及original_user_request，建立**全新**admin会话/任务并交给已运行Web的唯一FIFO；脚本不另启worker。可选case为cpu、wall、matrix、cover-a、cover-b，也可用 `--task-id` 指定已存在任务。新task强制 `dry_run=true`、`explicit_publish_requested=false`，不继承旧授权、回复ID、模型答案、摘要或历史文档。原目标与原图输入保留；图片失效时拒绝回放，不静默丢图。这个enqueue会让在线worker实际调用模型，不能当作离线准备命令。

回放记录在 `.aurex/cache/operator-replays/` 保存新task ID供中断核对。检查器将任何publication记录、社区replying/replied/unknown或实际reply receipt标为违规；没有ledger时返回“无法确认”，不假称零外发。检查范围是本地ledger，不代替社区侧人工核对。隔离测试 `.venv/bin/python scripts/aurex-operator-replay-selftest.py -v` 使用临时数据库并禁止真实HTTP/socket连接，14项通过；不代表这些旧任务已经实际重跑成功。

`prepare/enqueue --request-append '用户补充原话'` 可显式补充新的要求，同时追加到新task的两份请求字段；不改旧任务、原始上下文或历史消息。默认不追加时原文保持逐字节不变。追加存在性及原文/新请求SHA-256写入回执便于核对；空补充及超长请求拒绝执行。该脚本隔离测试现为23项。

新场景使用 `scripts/aurex-admin-scenario.py`：默认 `prepare --request '原始问题'` 只读；`enqueue --execute --request '原始问题' --receipt 新路径.json` 才加入现有FIFO。固定admin、dry-run、不发布、无社区回复目标，不能由请求正文覆盖。`--session-id` 可在现有admin会话继续提问，但每次新建task；不接受community会话。回执必须是新文件，提交前排他写入并同步到磁盘；不确定时只运行 `reconcile --receipt 路径.json`，不能重复提交。脚本不另启worker，20项模拟网络禁用测试覆盖这些边界。

## 请求与思考策略

每个用户问题先读取原帖标题、完整正文、评论、作者与时间、回复关系和原始分类/状态；封面和用户上传图片先归档为可读取的路径，不自动把像素送入模型。AI需要视觉信息时必须显式调用 `view_image` 或支持图片的工具并设置 `with_image=true`。每个用户问题的第一条全上下文模型请求传 `chat_template_kwargs.enable_thinking=true`；后面的工具调用、工具结果分析和候选答复整理传 `false`。提交发布前以及最终答复发出前分别进行独立thinking审核；这是独立证据审查，不是把整个工具循环改成思考模式。容器默认关闭思考，由客户端逐次设置。上下文摘要请求默认关闭思考。`reasoning_content` 只进入单独的跟踪事件，不作为 `assistant.content` 回放，也不写入公开作品介绍。

图片授权只属于当前task：旧任务的图片不会随会话历史自动重放，新任务需要重新调用工具。图像仍保存在Web与原始记录中，不因这项策略删除。元件值、节点连通性和仿真结果优先使用结构化数据；空间指代、封面内容等问题才按需看图，不以关键词硬禁用视觉。

工具使用标准 OpenAI `tool_calls` / `tool_call_id`，按实际结果继续执行。模型、工具或流中断时记录明确错误，禁止把截断或未完成仿真当成成功。原文、网页、评论和图片是参考资料，不能覆盖用户指令。

单次生成上限、上下文压缩策略和整个任务是否完成是不同问题。输出同时包含思考和正式结果；输入、图像、输出共同受实际模型窗口限制，截断不能作为任务完成条件。当前服务不自带固定 `max_new_tokens` 默认值：客户端不传输出上限时，服务端以90112减去实际输入长度作为这次生成的最大剩余空间，而不是无限输出。具体任务的未完成/续作记录保存在跟踪库。

## 视觉容器

```bash
bash scripts/aurex-vllm.sh start
bash scripts/aurex-vllm.sh status
bash scripts/aurex-vllm.sh stop
```

实际命令和测试边界见 [scripts/aurex-vllm.md](scripts/aurex-vllm.md)。部署采用 GPU 1、2，TP2，TurboMind AWQ、FLASH_ATTN_V100、2048 batched tokens、prefix caching 和 decode CUDA Graph `[1]`。视觉编码器使用 TORCH_SDPA；不设置 `--language-model-only`。

当前服务模型名 `qwen38-27b`，端点 `http://127.0.0.1:8000/v1`。上下文90112 tokens包括输入、图像和输出，每请求最多2张图，vLLM处理器限制单图像素面积不超过589824。应用先限制长边1024px，再由vLLM按该面积预算缩放：横向电路图可保留更多文字，仍不超过已验证的视觉profile。应用读取 `/v1/models` 校验实际上下文上限。`.945` 显存参数热身占用15.500GiB/卡，应用多轮请求后约15.65GiB；完整90K极限压力未实测。思考/非思考解码实测约45 tokens/s，采样期间两卡均达100%；这不是空闲或所有请求阶段持续100%的承诺。

## 安装与启动

```bash
cd /home/macromodel/Documents/src/aurex3
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/configure-aurex3.py
bash scripts/aurex-web.sh start
```

配置脚本从 `aurex3.config.example.json` 生成 `.config/aurex3.json`；配置及管理员访问码权限为0600，均被 Git 忽略。请在私有配置中填写社区账号，不要把账号密码或管理员访问码放进公开配置。已有文件不会被覆盖。

工作台默认绑定 `0.0.0.0:4097`，本机及 Tailscale 可访问 `http://100.123.133.75:4097`。管理员访问码在 `.config/web-token`。普通用户的数据按“浏览器能力凭据 + 关联用户 ID”隔离；裸用户 ID 不能跨设备恢复或接管会话。网页支持带图片提问、选择 Experiment/Discussion/User 目标、逐步查看工具结果和图片、下载本地实验文件。仍建议只在可信内网或 Tailscale 使用，不要把 HTTP 端口暴露到不可信公网。

```bash
# 首先在配置中选择 agent.dry_run；true只检查，不发社区评论。
bash scripts/aurex-web.sh start
bash scripts/aurex-web.sh status
bash scripts/aurex-web.sh stop

# 分区式终端界面；Web已经运行时直接连接同一数据库、队列和社区轮询器
PYTHONPATH=src .venv/bin/python -m aurex cli --config .config/aurex3.json

# 也可以直接运行（默认进入cli，默认配置为.config/aurex3.json）
PYTHONPATH=src .venv/bin/python -m aurex
```

CLI与Web是同一个Aurex v3前端：都连接唯一的常驻server、同一个持久化调度队列和社区轮询器。Web已运行时CLI直接附着；CLI先启动时会先拉起同一个后台Web server再附着，随后启动Web只会识别并复用它。因此两种启动顺序都不会创建第二个调度器，退出CLI也不会中断server或当前队列。终端按服务、任务队列、会话、当前任务时间线、输入区分区，`Tab`切换区域、方向键选择、`Enter`进入；输入`/help`可查看新会话、切换会话/任务、取消、显式发布和退出命令。旧版`chat`、`console`、`run --once`以及手动`--poll`入口均不存在。

通知轮询只响应显式 `@aurex`，默认新部署 `bootstrap_lookback_sec=0`，避免重答历史消息。Web、CLI、通知机器人和管理员API共用持久化调度队列；同一目标/提问者可归入同一会话，但每次用户提交都是独立task（`runs.id`），同一会话也不会把新问题合并成旧任务。默认 `max_parallel_tasks=1`，适配本机 TP2 单序列推理；其他部署可按后端能力增加不同会话的并发数。

管理员使用 `POST /api/tasks` 提交原始请求，可指定已有 `session_id`；不指定则建立新会话。普通用户使用 `POST /api/requests`。`GET /api/me` 返回当前角色及白名单公开资料；`GET /api/tasks`、`GET /api/tasks/:id` 和取消端点均由服务端执行所有权检查。HTTP客户端不能传 `source`、`purpose`、`metadata` 或自定task ID；`source=admin` 由管理员端点确定。

## 电路工具

| 工具 | 行为 |
|---|---|
| `plar_get_experiment_file` | 下载完整原始社区电学实验，保留Position/Rotation/CameraSave及原文，返回本地路径和SHA-256；不执行远程字符串 |
| `circuit_catalog` | 查询真实支持的元件、引脚顺序、SI参数和存档导出范围 |
| `circuit_inspect` | 读取 `.sav` / `.plsav` / `.circuit.json` / `.pe-state.json`，默认返回结构化数据；`with_image=true` 才渲染图片 |
| `circuit_create` | 添加模拟元件、连接命名节点并创建本地实验 |
| `circuit_edit` | 修改参数、添加/移除元件、连接端口；生成新版本，保留原稿 |
| `circuit_analyze` | 在隔离限时子进程调用真实 Phy-Engine DC、AC、瞬态及数字分析 |
| `circuit_read_stimulus` | 按元件和步骤分页读取已记录的数字激励及 L/H/X/Z 状态，不重新仿真 |
| `hdl_simulate` | 在离线受限沙箱编译/仿真单个或多个HDL源文件，返回源码哈希、真实日志和独立验证报告 |
| `verilog_to_sav` | 综合单个或多个Verilog源模块；可用成功HDL报告绑定同一份未改动源文件，生成存档、图像和导出清单 |
| `view_image` | 在本会话后续轮次重新读取已有电路图片 |
| `plar_publish_experiment` | 申请服务器授权范围内的独立发布审核；不能自行授权或指定封面，审核通过才进入持久化发布流程 |

渲染几何由 `third-parties/Phy-Engine/src/circuit_view.cpp` 实现，输出SVG，再用Cairo转换PNG。空间图保留原始坐标和Euler旋转：原始PLSAV优先使用实际 `CameraSave`；缺少可用保存相机时明确采用fallback等角视图，不伪造原始相机。蓝色元件体、端子、连线、短编号及简明数值帮助模型对应实际实验位置；元件外形和端子几何是示意图，不是物理实验室3D资源的逐像素复刻。只有缺失坐标才生成位置并标明来源。`view="topology"` 保留辅助拓扑图。

四个电路工具 `circuit_inspect/create/edit/analyze` 的 `with_image` 默认为严格布尔值 `false`，此时不执行SVG/PNG/Cairo渲染；显式 `true` 才生成图片。需要数字电路接口时优先调用 `circuit_inspect(interface_only=true)`，返回真实输入/输出的ID、Label、节点和已有逻辑状态，每页默认/最多64项，不展开内部门电路，也不求解。接口模式与 `with_image=true` 互斥，需另行请求图片。

显式请求大电路图片时先返回固定尺寸全局图；若远处元件令主体过小，同时返回明确标为“非全场景”的主体视图及离群元件信息，不移动或删除元件。`focus_ids`按原ID、图中编号或唯一原始Label定位目标，歧义Label报错而不猜测；`query`按类型/名称/属性查找，使用 `offset` / `limit` 和实际 `next_offset` 分页，邻居单独标记。局部观察建议每页4到8个元件；完整连接、原始Label与坐标保留在netlist和PE状态中。

普通验证请求优先从接口选取少量有代表性的输入并核对输出，不默认枚举全部组合或全部元件。报告必须说明实际样例、失败项和未覆盖范围；样例通过不是完整正确性证明。用户明确要求穷举或完整认证时才扩大验证范围。

局部图右下角预留全局定位图：复用完整场景位置，以灰蓝轮廓和采样连线显示大致布局，黄色标记主图中实际显示的元件。它包含远处离群点、不重复求解、不绘制引脚与密集标签；只是方位提示，不能代替精确连通性证据。固定角度的发布封面仍必须覆盖全部元件，不使用局部图替代。

两条相机入口使用相同的只读控制接口：

| 输入 | 数据来源 | 调整相机的影响 |
|---|---|---|
| 原始 `.sav` / `.plsav` | 原始元件、连线、位置/旋转、保存相机及存档状态 | 只改变视图，不修改或重新求解原实验；存档统计不等于新测量 |
| `circuit_analyze` 返回的 `.pe-state.json` | 原生电路spec、已求解samples与场景 | 直接渲染记录的引擎状态，不需要PLSAV中转；换相机不会重跑求解器 |

`camera.mode="saved"` 读取保存参数；`mode="auto", fit=true` 重新定帧；`mode="custom"` 可使用 `position+target` 或 `target+yaw_deg+pitch_deg+distance`，并设置zoom、FOV或正交投影。坐标是native xyz；PhysicsLab部分字符串按x,z,y存储，转换信息和保存的raw值会随结果返回。保存Euler的解释、缺失字段回退与默认FOV属于显式假设；`position+target` 使用确定的look-at与native +Z up，不声称所有历史文件的相机约定都完全一致。

模型应先查看 `camera` 中的来源、实际参数、裁切/behind-camera和假设，再结合图片决定是否调整视角或focus。**无裁切不等于无重叠**：密集标签或元件可能互相遮挡，可继续换角度、缩放或缩小focus；不能仅凭warnings为空宣称图像清晰。原始坐标和权威网表保持不变。

权威元件值与电气连通性来自存档/引擎数据；不会靠OCR猜阻值，线条交叉不代表电气连接。支持真实PL模型映射的设计才导出 `.sav`，其它原生模型保留 `.circuit.json` / `.pe-state.json` 并明确说明，不能伪造跨软件逐点仿真等价性。本地创建和分析本身不发布；对外发布使用下述独立授权与审核通路。

瞬态末态和实际时间序列是不同证据，不能从一个末态推断频率、占空比或持续振荡。通用原生 `analysis="tr"` 支持精确终点及 `tr_sample_every` 实际采样：时间单位为秒，最多10000个求解步、201个记录点，按指定步间隔和最终时刻读取真实状态，不插值伪造波形。新句柄从零节点电压/零电容历史开始，并非预先求出DC工作点；任意电容初始条件尚未暴露。现有电容为梯形伴随模型，起始瞬态和步长误差必须纳入判断。

NPN/PNP使用通用准静态双结Ebers–Moll模型，记录B/C/E端子电流（正号表示流入器件）。该模型覆盖正反向有源、饱和和截止，但没有结电容、存储时间、Early效应、雪崩、高注入或校准温漂。PL `Transistor` 导出保留B/C/E拓扑、极性和正向beta；其它PE物理参数留在metadata，不能声称原app逐点数值等价。二极管和MOS暂为native-only，缺少忠实映射时拒绝存档导出，不静默丢元件。详见 [原生模型与采样说明](third-parties/phy-engine-aurex-notes.md)。

通用原生观察测试13项、电路工具测试14项已通过，包含独立MNA电流/KCL核对、AC Jacobian有限差分、真实瞬态采样、精确终点、续算、取消与失败状态。这些是基础能力验证，不代表任意复杂电路或模型自主设计任务已通过。实际可用参数、引脚电流来源和模型局限仍以 `circuit_catalog` 及工具返回为准。

### HDL验证与绑定导出

`hdl_simulate` 支持调用方自建testbench的 `custom` 模式及服务器维护的固定验证profile。`custom.verified=true` 只表示该testbench完整运行成功，不证明测试覆盖充分。固定 `rv32i_teaching_v1` 验证明确列出的32位教学指令子集、接口及异常条件，不是完整RV32I认证或实际硬件时序验证；精确端口契约在工具描述中，不提供参考CPU实现。

Icarus Verilog运行于bubblewrap隔离文件系统/PID/网络，限制每阶段1GiB地址空间、20秒CPU时间、30秒墙钟时间及64KiB日志。用户源文件必须是有限大小的平面 `.v` / `.sv` 文件；文件/进程/VPI/DPI调用、包含文件及宏被拒绝，仅允许字面量 `timescale` 和有限安全系统任务。缺少沙箱时拒绝运行，不静默降级为无保护执行。部署可使用系统Icarus，或将受信任包解到 `.aurex/toolchains/iverilog-12.0-3/root`；操作员可用 `AUREX_IVERILOG_ROOT` 指向工具链根目录。

验证报告保存 `verification_id`、profile、每个源文件及集合SHA-256、编译/仿真退出状态与实际日志。将成功报告的 `report_path` 作为 `verilog_to_sav.hdl_report_path`，工具重新核对源哈希并导出同一份源文件，随后生成绑定RTL、验证报告和PLSAV哈希的 `.export.json`。源文件修改后必须重新验证；绑定导出不是RTL与PLSAV逐周期等价性的证明。

### 独立审核和至多一次发布

服务器按每个task持久化原始用户请求、来源、提问者及外发范围。每个task最多一次publication和一次最终答复；只有服务器认定原始请求明确要求发布（`explicit_publish_requested=true`），并且独立thinking审核也确认该意图，才可发布。模型提示、上传文件、旧会话授权标记或工具参数不能自行扩大权限。普通问答和未请求发布的设计任务只产生本地成果。旧两测试会话的 `authorize_session` 记录只作历史标记，不再单独授权发布；生产流程采用通用电学实验审核，不按测试提示哈希开启特殊通路。

发布请求必须提供新建PLSAV与真实验证证据。服务器先验证证据来源、完整结束状态、源/导出哈希及任务约束，再开启一次独立thinking审核；审核默认接收实际证据、源文件及固定封面生成清单，只有发布工具显式传 `with_image=true` 才向审核模型提供封面像素。封面仍然必须生成、验证并上传，图片开关不削弱该硬性规定。证据过大、互相矛盾、截断、源文件变动或审核不通过都不发布。正式标题和正文为中文，并说明已验证范围及局限，不含内部思考。

社区来源必须绑定真实提问者ID：发布正文末行与最终回复开头分别由服务器添加 `<user=ID>@nickname</user>`，不能让模型猜ID或改提问者。Web/管理员本地任务不伪造提问者、不添加该mention。最终答复的独立thinking审核可判定 `completed`、`blocked` 或 `continue`；`continue` 不占用最终答复次数，继续原任务的工具过程。`dry_run` 不产生社区外写。

正文mention与社区通知路由是两回事。`Messages/PostComment` JSON中的 `ReplyID` 必须是原提问者的用户ID，不能填原评论ID、实验ID或留言板主人ID。v3请求由持久化task绑定取得该ID，旧版兼容回复路径也使用评论作者ID；本地task的 `reply_id` 历史字段仍保存原评论ID用于去重，不直接作为HTTP ReplyID。新回执记录实际提交的目标和回复用户ID；API接受评论不等于已证明接收者设备显示通知。

通知发送采用已由接收者实测确认的官方 SDK 路径：内部审核答案仍保留 ID 型 mention，最终传输转换为 `回复@昵称: 正文`，显式传入绑定用户的 `reply_id`，由社区服务器生成显示用 ID 标签。不直接发送预渲染的 mention；单独删除版本头的旧路径实测未解决通知。SDK 在独立进程中单次调用，凭据仅经私有 stdin 传递，网络和进程均有超时保护；提交结果不确定时保留记录、禁止自动重发。回执记录服务端评论 ID、原审核文本和实际发送文本的 hash，`notification_status=unverified` 不把评论接受误报为每条通知均已送达。已有评论不会补发。

封面独立于模型的观察相机：服务器固定yaw=45°、pitch=60°、正交全场景fit，校验原实验全部元件已绘制且入框，并绑定原文件与封面哈希。模型不能选择封面视角、分页或focus来隐藏元件。全部入框仍可能互相遮挡，**不是每个元件或标签都可独立辨读**的保证。

`.aurex/cache/.publication/ledger.sqlite3` 持久化task范围、审核、不可变源/封面快照、发布及最终答复回执。创建提交采用**至多一次**语义：提交前先持久化状态，重启后不会重复创建；若提交结果不确定则保留unknown并要求人工核对，不能盲目重发。已取得社区ID后，封面上传/确认可从记录阶段继续；只有确认发布成功的回执才是发布成功证据。不要把这种流程说成外部服务的“恰好一次”保证。

构建输出在 `.aurex/cache/phy-engine-build`，需要支持本引擎的C++编译器与系统Cairo库。首次使用可以自动构建；大量编译应与vLLM装载错开，避免宿主RAM压力。内置源码已经包含 Aurex 所需改动，不需要额外应用补丁。`third-parties/phy-engine-aurex.patch` 只用于在独立的上游干净检出中审阅或复现历史差异。

补丁基线为 Phy-Engine `cdff6c17c3f7fb28154a8f3b776efad5cb0c8920`。当前内置源码已应用，不要在 Aurex 工作树内重复执行。仅在单独的 Phy-Engine 干净检出中复现时运行：

```bash
git apply --check /path/to/aurex3/third-parties/phy-engine-aurex.patch
git apply /path/to/aurex3/third-parties/phy-engine-aurex.patch
cmake -S third-parties/Phy-Engine/src -B .aurex/cache/phy-engine-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-21 \
  -DCMAKE_C_COMPILER=clang-21 -DPHY_ENGINE_USE_LEVELDB=OFF
cmake --build .aurex/cache/phy-engine-build \
  --target phyengine verilog2plsav circuit_view --parallel 1
```

## 搜索和证据

`web_search.provider` 可选 `auto`、`brave`、`searxng`、`bing`、`crossref`、`duckduckgo`。有 `BRAVE_SEARCH_API_KEY` 时优先Brave；配置 `base_url` 后可用自托管SearXNG。没有付费Key时保留匿名降级，过滤明显无关的Bing结果，并可查Crossref论文元数据。Crossref结果不是论文全文；没有配置的搜索服务不会假装可用。

`web_fetch` 按大小/超时限制读取网页，检查重定向与公网地址；兼容本机Clash fake-IP时仍固定真实公网IP，禁止读取loopback、Tailscale和内网地址。完整网页和工具返回可归档后按页回读。

## 长上下文与持久化

上下文管理不是任务总token或工具轮数预算。v3任务不会因为沿用旧 `agent.max_tool_loops` 达到固定轮数而收尾；`llm.max_output_tokens=null` 默认省略单请求输出上限，交给服务器的真实剩余窗口决定。单次响应 `length` 时归档不完整正式输出/工具参数，**不执行半截工具调用**，并继续原任务；网络故障、非法响应、用户取消与真正阻碍仍明确记录。授权发布任务没有成功回执时保持 `needs_attention`，不能只因模型说“完成”就标成功。

重复的确定性结果、方案来回切换及改变步长后仍出现的同类仿真失败会触发策略复核：先用no-think轮整理证据草稿，再由独立thinking审核判断继续、完成或确有阻碍。它不跳过工具执行、不把失败视为测量结果，也不按次数结束任务；审核 `continue` 后恢复工具并清空本轮重复检查窗口，成功仿真也会清空失败窗口。

借鉴OpenCode的预算与压缩机制：预留输出和安全空间，优先调用vLLM tokenizer计算真实输入长度；接近阈值时压缩旧对话、保留完整工具调用/结果对及最近内容。长原文先分块摘要，原文与工具全文始终保存在SQLite，`read_context` 可按document_id分页恢复。不会把所有历史思考重新塞入模型。

这些行为由顶层 `context` 配置，而非写死；完整结构与可选示例见 [示例配置](aurex3.config.example.json)。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `auto_compact` | `true` | 允许自动摘要/checkpoint；关闭后不会静默摘要，不能装入窗口时明确报错并保留原文 |
| `prune` / `prune_keep_tool_results` | `false` / `4` | 超阈值后可把较旧工具正文替换成可回读引用，保留最近若干结果；不删数据库原文 |
| `reserved_output_tokens` / `safety_tokens` | `null` / `2048` | 输入侧输出余量与安全余量，不是任务总预算，也不直接限制实际生成长度；输出余量不得小于显式单请求输出上限 |
| `compact_at_ratio` | `null` | 继承 `llm.compact_at_ratio`（默认0.8），或显式设置相对于可用输入空间的整理阈值 |
| `document_budget_ratio` / `tool_output_tokens` | `0.4` / `null` | 长文档/工具正文的单块预算；工具值为空时沿用文档比例 |
| `retain_recent_turns` / `retain_recent_tokens` | `1` / `null` | checkpoint的近期保留偏好；切分仍必须保持完整工具调用/结果配对 |
| `summary_max_tokens` / `summary_thinking` | `4096` / `false` | 摘要请求独立参数，不改变用户首轮或发布审核的thinking策略 |
| `prune_images` | `true` | 只为当前请求保留最近的 `llm.max_images` 张图，旧图仍可回读；关闭时超图数明确报错 |
| `profile` / `profiles` | `null` / `{}` | 管理员按名称选择一组平面字段覆盖；不由模型自选，也不能突破服务端真实窗口 |

未显式设置 `reserved_output_tokens` 时，client按真实窗口派生输入侧余量；当前默认派生量为窗口的四分之一。该配置是最小预留量，可增加保守余量，但不能将余量降到client派生值以下。默认自动摘要与可选工具正文裁剪相互独立。示例中的profile只展示配置方法，不会自动启用；不论profile如何选择，真实窗口、图片上限、完整原文持久化及错误标记保持有效。

`llm.reasoning_effort` 默认 `null`，请求不传该字段而保留模板默认；可选 `low` / `medium` / `xhigh`。当前Qwen模板默认xhigh，medium仍开启思考而不额外注入xhigh提示，low要求简短思考；这是模板文本控制，不是硬token配额，也不是质量保证。它与 `enable_thinking`、上下文整理策略分别配置。

`.aurex/aurex.sqlite3` 使用SQLite WAL，持久保存每个会话、排队请求（含图片路径）、原始消息、工具事件、原文、摘要和文件索引。Web重启会恢复排队请求；运行中被中断的请求标记 interrupted，并补齐工具调用的失败结果，以免恢复时协议不完整。已经执行但未返回的工具副作用视为未知，需要查现有文件后再续做。不会清空旧会话。

磁盘历史保留与上面的模型上下文压缩互不相同。`storage.history_dir` 只管理该专用目录的直属备份快照，默认留空时从当前 `tracking.database_path` 推导为同目录的 `backups/`，避免测试数据库误碰生产历史；示例部署显式使用 `.aurex/backups`。服务启动后立即检查，之后默认每300秒检查一次：目录实际占用达到10 GiB时，按最旧优先把稳定超过5分钟的原始快照逐份原子压缩为 `.tar.gz`，逐文件校验内容一致后才删除源快照；压缩失败或快照在压缩期间变化时保留原件。压缩后仍达到20 GiB时，才按最旧优先删除已验证压缩包，刚降到20 GiB以下即停止，并始终保留最新一份快照。每个工作单元只处理一份快照，超水位时连续收敛，关服信号会在快照边界停止后续处理。

保留器不会扫描或删除当前SQLite数据库/WAL、`.aurex/cache`、任务artifact，也不会触碰运行、排队或其他未完成任务的数据；历史目录与归档分别强制为 `0700`、`0600`。默认数据库同级 `backups/` 会自动初始化专用标记；其他自定义非空目录没有该标记时会拒绝启用，包含数据库/缓存、符号链接或挂载边界的快照也不会处理，防止误配普通目录。自动删除不可恢复，重要历史仍应另存独立备份。阈值和检查周期可在 `storage` 中配置：`history_retention_enabled`、`history_compress_at_gib`、`history_delete_at_gib`、`history_check_interval_sec`、`history_min_age_sec`，其中删除阈值必须不小于压缩阈值。

流式推理使用可取消的异步HTTP读取，保持同步agent接口。工具解析器尚未吐出SSE不等于模型停止：不再用固定180秒静默读取超时结束任务。主线程每0.5秒检查取消；静默30秒后通过独立只读 `/v1/models` 做健康检查，每次最多5秒，连续3次失败才报告失联，期间新的SSE优先。健康探测不会重试或新建推理请求，也不能证明某条任务正在取得进展；服务健康但任务停滞时仍可人工取消。取消会关闭活动HTTP流，真正断流、缺少finish/`[DONE]`、不完整工具调用均保留为错误，而不是成功。

## 验证

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q
AUREX_PHY_ENGINE_BUILD="$PWD/.aurex/cache/phy-engine-build" \
  PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_circuit_tools.py -v
.venv/bin/python scripts/aurex3-smoke.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_hdl.py -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_publication_review.py -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_plar_publish.py -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_stream_lifecycle.py -v
.venv/bin/python scripts/aurex-design-e2e-selftest.py -v
PYTHONPATH=src AUREX_PHY_ENGINE_BUILD=.aurex/cache/phy-engine-build \
  .venv/bin/python -m unittest discover -s tests -p test_native_observation.py -v
```

关键回归覆盖首轮思考/工具轮no-think、思考不回放、流截断不能成功、图片与工具结果进入下一请求、超长输入归档、队列和图片恢复、工具调用中断修复、会话文档隔离，以及网页切换/上传期间的异步竞态。原有v2配置和测试保留兼容路径；v3使用 `llm.enabled=true`。

流生命周期19项测试使用本地模拟HTTP服务，不调用真实模型：覆盖工具参数缓冲超过旧读取时限、等待响应头/响应体/健康探测时取消、连接关闭与线程清理、健康失联和完整/不完整工具结果。这证明传输边界行为，不等于重跑过自主电路设计。

复杂电路相机已完成真实Qwen双入口测试：模型分别读取保存相机和native状态，自主调整custom camera，并在第二条任务中再次换角度改善重叠；原始文件哈希不变，无重复求解。Web图片、思考/工具折叠、延迟会话切换均通过。详细记录保留不足而不只列成功，见运行机器 `.aurex/cache/camera-validation/assessment.md`。

HDL安全回归和发布/独立审核回归不等于真实自主设计发布已经通过。实时端到端设计结果分别保存在 `.aurex/cache/design-e2e/`，预算耗尽、传输/仿真/导出失败和未完成状态也保留；只有实际日志和发布回执支持时才报告成功。

`scripts/aurex-design-e2e.py prepare` 只准备自然任务目标，不创建任务。原 `--case riscv` / `--case 555` 保留用户明确的发布要求；[13题盲测矩阵](scripts/aurex-blind-tasks.json) 全部默认不发布，也不自动运行。`run --execute` 一次只提交一个管理员任务，可选 `--session` 复用对话，但服务器始终分配新的task ID。脚本不调用授权函数、不发送内部purpose、不把独立verifier或期望答案塞入提示。

```bash
# 离线准备；这条命令不调用模型
.venv/bin/python scripts/aurex-design-e2e.py prepare
# 只有操作员明确要提交时才运行以下命令；每次都是新任务
.venv/bin/python scripts/aurex-design-e2e.py run --blind-task digital-adder --execute
# 查询或继续观察原任务，不再次提交，也不取消它
.venv/bin/python scripts/aurex-design-e2e.py status --task-id TASK_ID
.venv/bin/python scripts/aurex-design-e2e.py observe --task-id TASK_ID --timeout 1800
```

上下文题必须由 `--input-file` 提供真实输入，JSON只接受 `reference_text`、`images`、`target`；其中原文以参考资料而非指令附加，图片与目标交给既有API。不要把答案、oracle或验收说明当输入。`--timeout` 只限制观察客户端的等待，不是任务预算；等待到期或断网只归档原task ID，绝不自动重发。创建请求超时后若未拿到明确回执，记录 `unknown_do_not_resubmit`，先只读核对 `/api/tasks`。服务状态completed也只记为 `assessment=not_scored`，最终验收还需独立证据。脚本15项隔离自测全局禁止真实HTTP/socket连接，验证新task、旧任务续查、未知回执、不发布默认和oracle隔离。
