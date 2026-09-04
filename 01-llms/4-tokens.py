"""Tokens are not words.

count_tokens asks the API how a string breaks up for a given model.

Do NOT use tiktoken here. That is OpenAI's tokenizer, and it is simply
wrong for Claude — it undercounts by 15-20% on plain English, and by far
more on code or on any language that is not English.

Note the floor: every request carries a little structural overhead, so
even a one-word message costs more than one token.

    python 4-tokens.py
"""

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"

SAMPLES = [
    "hello",
    "hello world",
    "antidisestablishmentarianism",
    "def f(x): return x * 2",
    "שלום עולם",
    "🙂🙂🙂",
]

print(f"{'tokens':>6}  {'words':>5}  {'chars':>5}   text")
for text in SAMPLES:
    n = client.messages.count_tokens(
        model=MODEL,
        messages=[{"role": "user", "content": text}],
    ).input_tokens
    print(f"{n:6d}  {len(text.split()):5d}  {len(text):5d}   {text!r}")

# Long English words split into several tokens.
# Code splits on punctuation.
# Non-English text and emoji cost far more tokens per character.
