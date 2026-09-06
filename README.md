# ai-material

Code and material for **AI Agents for Developers**:
https://www.yuval.guide/courses/ai-agents/

One directory per chapter. Each lesson tells you which file to run.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo 'export ANTHROPIC_API_KEY=sk-ant-...' > .env
source .env
```

`.env` is ignored by git, so the key stays on your machine. In every new
terminal, activate the venv and `source .env` again. On Windows, skip the
`.env` lines and run `setx ANTHROPIC_API_KEY sk-ant-...` once instead.

Every script here costs a fraction of a cent to run. `01-llms/5-cost.py`
prints exactly how much.
