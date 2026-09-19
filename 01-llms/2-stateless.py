"""The model has no memory.

A chat prompt that is not a chat. Every line you type is sent as its own
call, carrying nothing but that one line. Talk to it for as long as you
like: it will never know what you said a moment ago.

Each turn prints the request and the reply the way the API sees them — a
role and a block type per line — so you can watch the request stay exactly
one message long, turn after turn.

    python 2-stateless.py

Type /quit, or press Ctrl-D, to leave.
"""

import textwrap

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"

# Sent on every call, and the only thing that is. It keeps the answers
# short, which keeps this lab cheap.
SYSTEM = "You are chatting with a developer. Answer in one or two short sentences."

# Low effort: a short chat answer needs no deliberation, and effort is part
# of what you pay for. See 5-cost.py.
EFFORT = "low"

WIDTH = 72


def line(role, block_type, text, clip=False):
    """One message part, labelled the way the API labels it.

    A reply wraps under its own label; a thinking summary is clipped to one
    line, because its length is not the point.
    """
    text = " ".join(text.split())
    label = f"  {role:<9} {block_type:<8} "
    if clip:
        if len(text) > WIDTH:
            text = text[: WIDTH - 1] + "…"
        print(label + text)
        return
    for i, part in enumerate(textwrap.wrap(text, WIDTH) or [""]):
        print((label if i == 0 else " " * len(label)) + part)


def ask(question):
    """One call. One message. No history, because there is nowhere to keep it."""
    messages = [{"role": "user", "content": question}]

    line("user", "text", question)

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": EFFORT},
        messages=messages,
    )

    for block in response.content:
        if block.type == "thinking" and block.thinking:
            line("assistant", "thinking", block.thinking, clip=True)
        elif block.type == "text":
            line("assistant", "text", block.text)

    print(
        f"  {'':<9} {'':<8} "
        f"sent {len(messages)} message, {response.usage.input_tokens} input tokens"
    )


print("Every line you type is a new call. Nothing from earlier turns goes with it.")
print("Try: tell it your name, ask for it back, then ask again in other words.")
print("/quit to leave.")
print()

turns = 0
while True:
    try:
        question = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not question:
        continue
    if question in ("/quit", "/exit"):
        break

    ask(question)
    turns += 1
    print()

print(f"{turns} turns, {turns} calls, and every one of them sent a single message.")
print("The memory you are used to is something a harness sends. See 3-conversation.py.")
