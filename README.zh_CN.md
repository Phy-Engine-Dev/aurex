# aurex

本仓库包含一个面向工程化使用的物理实验室 AR 自动化 Agent，位于 `src/phy_lab`。

## phy_lab 快速开始

运行环境：

- Python 3.10+
- 本机运行的 Ollama（默认 `http://127.0.0.1:11434`）

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r src/phy_lab/requirements.txt
```

创建配置：

```bash
python src/phy_lab/agent.py init --config .phy_lab/config.json
```

运行（默认密码运行时输入，不会写入磁盘；测试场景也可在 `account.password` 写入配置）：

```bash
python src/phy_lab/agent.py run --config .phy_lab/config.json
```

安全试跑（单次轮询、不发帖）：

```bash
python src/phy_lab/agent.py run --config .phy_lab/config.json --once --dry-run
```

完整文档：

- 英文：`src/phy_lab/README.md`
- 简体中文：`src/phy_lab/README.zh_CN.md`
