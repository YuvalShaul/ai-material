"""Tool calling is a protocol, not a feature.

The model cannot run anything. What it can do is write a request — a
tool_use block naming a tool and its arguments — and then STOP. Your code
runs the tool and sends the result back as the next message.

This file makes the two calls by hand, with no loop, so you can watch the
protocol happen one message at a time.

    python 1-protocol.py
"""

import json
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"

STOCK = {"SKU-101": 12, "SKU-102": 0, "SKU-103": 4}


def lookup_stock(sku: str) -> int:
    return STOCK.get(sku, 0)


# What the model sees. The description and the schema are the whole
# specification — it decides from these whether to ask for the tool.
TOOLS = [{
    "name": "lookup_stock",
    "description": "How many units of one SKU are in the warehouse right now.",
    "input_schema": {
        "type": "object",
        "properties": {"sku": {"type": "string", "description": "The SKU code, e.g. SKU-101"}},
        "required": ["sku"],
    },
}]

messages = [{"role": "user", "content": "How many units of SKU-102 do we have?"}]

# --- call 1: the model asks for a tool, then halts ---------------------
r = client.messages.create(model=MODEL, max_tokens=1000, tools=TOOLS, messages=messages)

print("--- call 1 ---")
print("stop_reason:", r.stop_reason)
for block in r.content:
    if block.type == "tool_use":
        print(f"tool_use   : {block.name}({json.dumps(block.input)})  id={block.id}")
    elif block.type == "text":
        print(f"text       : {block.text}")

call = next(b for b in r.content if b.type == "tool_use")

# The model's turn goes into the history exactly as it came back...
messages.append({"role": "assistant", "content": r.content})

# ...then YOUR code runs the tool, and reports the result under the same id.
result = lookup_stock(**call.input)
messages.append({"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": call.id, "content": str(result)},
]})
print(f"you ran    : lookup_stock({json.dumps(call.input)}) -> {result}")

# --- call 2: the model has the result, and answers ----------------------
r = client.messages.create(model=MODEL, max_tokens=1000, tools=TOOLS, messages=messages)

print()
print("--- call 2 ---")
print("stop_reason:", r.stop_reason)
print("".join(b.text for b in r.content if b.type == "text"))

# stop_reason was "tool_use", then "end_turn". That one field is the
# exit condition of the loop you write in the next file.
