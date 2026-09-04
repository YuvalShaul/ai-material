"""The same task, asked two ways.

Nothing here is a trick. The second prompt simply says what the first one
assumed you would infer: the job, the shape of the answer, and the limits.

    python 1-vague-vs-specific.py
"""

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"

LOG = """
2026-09-04 11:02:14 ERROR db.pool timeout acquiring connection (waited 30s)
2026-09-04 11:02:14 ERROR api.orders 500 GET /orders/8812
2026-09-04 11:02:19 ERROR db.pool timeout acquiring connection (waited 30s)
2026-09-04 11:03:41 WARN  db.pool pool exhausted: 20/20 in use
2026-09-04 11:05:02 INFO  db.pool recovered, 3/20 in use
"""

VAGUE = f"Look at these logs.\n{LOG}"

SPECIFIC = f"""Here are five log lines from one service.

Say what went wrong, in at most three sentences.
Name the component and the time window.
Do not suggest fixes. Do not restate the log lines.

{LOG}"""


def ask(prompt):
    r = client.messages.create(
        model=MODEL, max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, r.usage.output_tokens


for label, prompt in [("VAGUE", VAGUE), ("SPECIFIC", SPECIFIC)]:
    text, out = ask(prompt)
    print("=" * 60)
    print(label, f"({out} output tokens)")
    print("=" * 60)
    print(text.strip())
    print()

# The vague prompt is not answered badly. It is answered for a different
# question than the one you had in mind, and it costs more tokens doing it.
