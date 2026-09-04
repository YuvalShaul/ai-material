"""Compression, written out rather than pressed as a button.

/compact in Claude Code did this for you. Here it is as fifteen
lines, so you can see what it keeps and what it throws away.

The rule: keep the last few turns whole, replace everything older with a
summary the model writes.

    python 5-compress.py
"""

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"
KEEP_RECENT = 2          # turns kept verbatim; everything older is folded

QUESTIONS = [
    "In two sentences: what is a database index?",
    "And what does it cost to keep one?",
    "When would you not add one?",
    "How does that change for a write-heavy table?",
    "What was the exact cost I asked about in my second question?",
]


def say(messages, max_tokens=400):
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, messages=messages)
    return "".join(b.text for b in r.content if b.type == "text"), r.usage


def compress(messages):
    """Fold everything but the last KEEP_RECENT turns into one summary."""
    keep = messages[-KEEP_RECENT * 2:]
    old = messages[: len(messages) - len(keep)]
    if not old:
        return messages
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
    summary, _ = say(
        [{"role": "user",
          "content": "Summarise this conversation in under 80 words. Keep "
                     "decisions and facts; drop pleasantries.\n\n" + transcript}],
        max_tokens=300)
    return [{"role": "user", "content": f"[earlier conversation, summarised]\n{summary}"},
            {"role": "assistant", "content": "Understood."}] + keep


for mode in ("full history", "compressed"):
    messages, sent = [], 0
    for q in QUESTIONS:
        if mode == "compressed" and len(messages) > KEEP_RECENT * 2:
            messages = compress(messages)
        messages.append({"role": "user", "content": q})
        answer, usage = say(messages)
        messages.append({"role": "assistant", "content": answer})
        sent += usage.input_tokens
    print(f"{mode:>14}: {sent:>6} input tokens over {len(QUESTIONS)} turns")
    print(f"{'last answer':>14}: {answer.strip()[:150]}")
    print()

# Read the last answer in each run. The compressed one is cheaper, and it
# may no longer be able to quote your second question back to you. That is
# the trade, and nothing warned you about it at the time.
