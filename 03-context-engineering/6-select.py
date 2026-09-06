"""Select: send what the question needs, not what you happen to have.

Builds a small directory, then answers one question about it twice —
once by sending every file, once by sending the two that matter.

    python 6-select.py
"""

import pathlib
import tempfile
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"

FILES = {
    "README.md":     "# billing\n\nInvoices and payment runs. Start with `python app.py`.\n" + "Background prose.\n" * 40,
    "app.py":        "import settings   # project settings first: they override module defaults\nimport payments\nimport invoices\n\n\ndef main():\n    ...\n" + "# filler\n" * 30,
    "payments.py":   "RETRY_LIMIT = 3\n\ndef charge(card, amount):\n    for attempt in range(RETRY_LIMIT):\n        ...\n" + "# filler\n" * 40,
    "invoices.py":   "def render(invoice):\n    ...\n" + "# filler\n" * 60,
    "settings.py":   "TIMEOUT = 30\nRETRY_LIMIT = 5   # overrides payments.py\n\nimport payments\npayments.RETRY_LIMIT = RETRY_LIMIT   # applied at startup, when app.py imports settings\n" + "# filler\n" * 30,
    "test_utils.py": "def fixture():\n    ...\n" + "# filler\n" * 50,
}

QUESTION = "What is the effective retry limit?"

d = pathlib.Path(tempfile.mkdtemp())
for name, body in FILES.items():
    (d / name).write_text(body)


def ask(context):
    r = client.messages.create(
        model=MODEL, max_tokens=500,
        messages=[{"role": "user", "content": f"{context}\n\n{QUESTION}"}])
    return "".join(b.text for b in r.content if b.type == "text"), r.usage.input_tokens


# --- everything -------------------------------------------------------
everything = "\n\n".join(f"--- {p.name} ---\n{p.read_text()}" for p in sorted(d.iterdir()))
a1, in1 = ask(everything)

# --- only the files that mention the thing asked about -----------------
hits = [p for p in sorted(d.iterdir()) if "RETRY_LIMIT" in p.read_text()]
selected = "\n\n".join(f"--- {p.name} ---\n{p.read_text()}" for p in hits)
a2, in2 = ask(selected)

print(f"whole directory : {in1:>6} input tokens  ({len(FILES)} files)")
print(f"grep first      : {in2:>6} input tokens  ({len(hits)} files: {', '.join(p.name for p in hits)})")
print(f"saved           : {100 * (in1 - in2) // in1:>5}%")
print()
print("--- answer from the selected files ---")
print(a2.strip())

# Selection is not only cheaper. The second window has nothing in it that
# could distract from the question. Chapter 6 is this, industrialised.
