"""The agent loop, in about forty lines of plain Python.

A messages list, a while loop, a dispatch table and one exit condition.
Every agent you have used is this shape.

Token counts are printed on every turn, so the cost of the loop is
something you watch, not something you find out at the end of the month.

    python 2-loop.py
"""

import anthropic

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
def run_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]        # the memory
    turn = 0
    while True:                                               # the loop
        turn += 1
        r = client.messages.create(model=MODEL, max_tokens=1000, tools=TOOLS, messages=messages)
        print(f"turn {turn}: sent {r.usage.input_tokens:>5} tokens, "
              f"got {r.usage.output_tokens:>4}, stop_reason={r.stop_reason}")
        messages.append({"role": "assistant", "content": r.content})

        calls = [b for b in r.content if b.type == "tool_use"]
        if not calls:                                         # the exit condition
            return "".join(b.text for b in r.content if b.type == "text")

        results = []
        for call in calls:                                    # the dispatch
            output = DISPATCH[call.name](**call.input)
            print(f"        {call.name}({call.input}) -> {output}")
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": str(output)})
        messages.append({"role": "user", "content": results})
# -------------------------------------------------------------------------


if __name__ == "__main__":
    answer = run_agent(
        "Which of SKU-101, SKU-102 and SKU-103 need reordering? Anything under 5 units does."
    )
    print()
    print(answer)
