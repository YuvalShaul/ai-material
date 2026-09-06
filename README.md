# ai-material

Code and material for **AI Agents for Developers**:
https://www.yuval.guide/courses/ai-agents/

One directory per chapter. Each lesson tells you which file to run.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo 'export ANTHROPIC_API_KEY=sk-ant-...' > .env
source .env
```

`.env` is ignored by git, so the key stays on your machine. In every new
terminal, activate the venv and `source .env` again.

The commands are for bash: Linux and macOS as they are; on Windows, inside
WSL 2. The course's introduction explains the setup.

Every script here costs a fraction of a cent to run. `01-llms/5-cost.py`
prints exactly how much.
