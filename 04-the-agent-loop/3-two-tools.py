"""Two tools, one real task.

Chapter 3's select lab built a small project and you grepped it by hand
to find which file decides RETRY_LIMIT. This time the agent holds the
grep and the read, and decides for itself which files are worth paying
for. Same files, same question.

The loop is the one from 2-loop.py, with a turn cap added: six lines,
marked below.

    python 3-two-tools.py
    python 3-two-tools.py "Which file defines the charge function, and what does it take?"
"""

import pathlib
import sys
import tempfile
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"
MAX_TURNS = 10            # <- the bound

FILES = {
    "README.md":     "# billing\n\nInvoices and payment runs.\n" + "Background prose.\n" * 40,
    "payments.py":   "RETRY_LIMIT = 3\n\ndef charge(card, amount):\n    ...\n" + "# filler\n" * 40,
    "invoices.py":   "def render(invoice):\n    ...\n" + "# filler\n" * 60,
    "settings.py":   "TIMEOUT = 30\nRETRY_LIMIT = 5   # overrides payments.py\n" + "# filler\n" * 30,
    "test_utils.py": "def fixture():\n    ...\n" + "# filler\n" * 50,
}
ROOT = pathlib.Path(tempfile.mkdtemp())
for name, body in FILES.items():
    (ROOT / name).write_text(body)


# ---- the tools ----------------------------------------------------------
def search_files(pattern: str) -> str:
    hits = []
    for path in sorted(ROOT.iterdir()):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if pattern in line:
                hits.append(f"{path.name}:{n}: {line}")
    return "\n".join(hits) or "no matches"


def read_file(name: str) -> str:
    path = ROOT / name
    return path.read_text() if path.exists() else f"no such file: {name}"


TOOLS = [
    {
        "name": "search_files",
        "description": "Search every file in the project for a string. Returns file:line: text "
                       "for each match. Cheap — use it before reading a whole file.",
        "input_schema": {"type": "object",
                         "properties": {"pattern": {"type": "string"}},
                         "required": ["pattern"]},
    },
    {
        "name": "read_file",
        "description": "Return the full contents of one file in the project. "
                       "Costs tokens for every line, so read only files a search pointed at.",
        "input_schema": {"type": "object",
                         "properties": {"name": {"type": "string", "description": "file name"}},
                         "required": ["name"]},
    },
]
DISPATCH = {"search_files": search_files, "read_file": read_file}


# ---- the engine, from 2-loop.py, plus the bound -------------------------
def run_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    total_in = total_out = 0
    turn = 0
    while True:
        turn += 1
        if turn > MAX_TURNS:                                  # <- the bound
            raise RuntimeError(f"no answer after {MAX_TURNS} turns")
        r = client.messages.create(model=MODEL, max_tokens=1000, tools=TOOLS, messages=messages)
        total_in += r.usage.input_tokens                      # bookkeeping
        total_out += r.usage.output_tokens
        print(f"turn {turn}: sent {r.usage.input_tokens:>5} tokens, "
              f"got {r.usage.output_tokens:>4}, stop_reason={r.stop_reason}")
        messages.append({"role": "assistant", "content": r.content})

        calls = [b for b in r.content if b.type == "tool_use"]
        if not calls:
            print(f"total: {total_in} tokens sent, {total_out} received, {turn} turns")
            return "".join(b.text for b in r.content if b.type == "text")

        results = []
        for call in calls:
            output = DISPATCH[call.name](**call.input)
            shown = output if len(output) < 70 else f"{len(output.splitlines())} lines"
            print(f"        {call.name}({call.input}) -> {shown}")
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": output})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "What is the effective retry limit, and which file wins?"
    print(run_agent(question))
