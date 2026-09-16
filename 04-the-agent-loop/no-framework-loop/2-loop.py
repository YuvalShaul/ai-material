"""The agent loop, in about forty lines of plain Python.

A messages list, a while loop, a dispatch table and one exit condition.
Every agent you have used is this shape.

Every turn prints the list it sends and the reply it gets, with the token
counts, so the cost of re-sending the whole conversation is something you
watch rather than something you meet at the end of the month.

    python 2-loop.py
"""

import anthropic

from wire_view import print_messages, print_reply

client = anthropic.Anthropic()
MODEL = "claude-opus-5"

STOCK = {"SKU-101": 12, "SKU-102": 0, "SKU-103": 4}


def lookup_stock(sku: str) -> int:
    return STOCK.get(sku, 0)


TOOLS = [{
    "name": "lookup_stock",
    "description": "How many units of one SKU are in the warehouse right now.",
    "input_schema": {
        "type": "object",
        "properties": {"sku": {"type": "string", "description": "The SKU code, e.g. SKU-101"}},
        "required": ["sku"],
    },
}]
DISPATCH = {"lookup_stock": lookup_stock}


# ---- the engine ---------------------------------------------------------
# Called, not run, when Python reads this file, so the helpers it uses may
# be defined further down.
def run_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]        # the memory
    turn = 0
    while True:                                               # the loop
        turn += 1
        reply = call_model(f"turn {turn}", messages)
        messages.append({"role": "assistant", "content": reply.content})

        calls = collect_tool_calls(reply)
        if not calls:                                         # the exit condition
            return reply_text(reply)

        results = []
        for call in calls:                                    # the dispatch
            output = DISPATCH[call.name](**call.input)
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": str(output)})
        messages.append({"role": "user", "content": results})
# -------------------------------------------------------------------------


def call_model(title, messages):
    """One request and one reply: the messages sent up, and the one that comes back."""
    print(f"=== {title} ===")
    print_messages(messages)
    reply = client.messages.create(model=MODEL, max_tokens=1000, tools=TOOLS, messages=messages)
    print_reply(reply, tokens=True)
    return reply


def collect_tool_calls(reply):
    """The tool_use blocks in a reply, in the order they arrived."""
    calls = []
    for block in reply.content:
        if block.type == "tool_use":
            calls.append(block)
    return calls


def reply_text(reply):
    """The text blocks of a reply, joined into the answer."""
    parts = []
    for block in reply.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)


if __name__ == "__main__":
    answer = run_agent(
        "Which of SKU-101, SKU-102 and SKU-103 need reordering? Anything under 5 units does."
    )
    print(answer)
