"""What a conversation costs, turn by turn.

Chapter 1 showed that a conversation is a list you re-send. This measures
what that costs once the list gets long.

Watch the input tokens. You are not paying for your question; you are
paying for every question and answer before it.

    python 4-cost-of-history.py
"""

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"
IN_PRICE, OUT_PRICE = 5.00, 25.00        # USD per 1M tokens, Claude Opus 5

QUESTIONS = [
    "In two sentences: what is a database index?",
    "And what does it cost to keep one?",
    "When would you not add one?",
    "How does that change for a write-heavy table?",
    "Summarise the whole thread in one sentence.",
]

messages = []
total_in = total_out = 0

print(f"{'turn':>4} {'sent':>8} {'back':>7} {'$ so far':>10}")
for i, q in enumerate(QUESTIONS, 1):
    messages.append({"role": "user", "content": q})
    r = client.messages.create(model=MODEL, max_tokens=400, messages=messages)
    answer = "".join(b.text for b in r.content if b.type == "text")
    messages.append({"role": "assistant", "content": answer})

    total_in += r.usage.input_tokens
    total_out += r.usage.output_tokens
    cost = total_in / 1e6 * IN_PRICE + total_out / 1e6 * OUT_PRICE
    print(f"{i:>4} {r.usage.input_tokens:>8} {r.usage.output_tokens:>7} {cost:>10.5f}")

print()
print(f"the last turn sent {r.usage.input_tokens} tokens to ask "
      f"{len(QUESTIONS[-1].split())} words")
