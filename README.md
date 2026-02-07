# aurex

This repository includes a production-oriented Physics Lab AR automation agent under `src/phy_lab`.

## phy_lab Quick Start

Prerequisites:

- Python 3.10+
- Ollama running locally (default: `http://127.0.0.1:11434`)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r src/phy_lab/requirements.txt
```

Create config:

```bash
python src/phy_lab/agent.py init --config .phy_lab/config.json
```

Run (password is prompted at runtime and never stored):

```bash
python src/phy_lab/agent.py run --config .phy_lab/config.json
```

Safe trial:

```bash
python src/phy_lab/agent.py run --config .phy_lab/config.json --once --dry-run
```

Full documentation:

- English: `src/phy_lab/README.md`
- Simplified Chinese: `src/phy_lab/README.zh_CN.md`

