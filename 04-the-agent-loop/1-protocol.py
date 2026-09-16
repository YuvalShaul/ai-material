"""
Demonstrating the model-agent wire protocol, using tool calling.

- The model cannot run anything.
- What it can do is write a request — a tool_use block naming a tool and its arguments.
- The model then stops generating text.
- Your code runs the tool and sends the result back as the next message.

This file makes the two calls by hand, with no loop, so you can watch the
protocol happen one message at a time.

    python 1-protocol.py
"""

import json
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"

# These are exchange stock codes
STOCK = {"SKU-101": 12, "SKU-102": 0, "SKU-103": 4}


def lookup_stock(sku: str) -> int:
    return STOCK.get(sku, 0)


# - A list of tools sent to the model, with one tool in it.
# - The schema tells the model which properties to send with the tool, in this
#   case a single property called sku (a string, the SKU code).
# The model decides from these whether to ask for the tool.
TOOLS = [{
    "name": "lookup_stock",
    "description": "How many units of one SKU are in the warehouse right now.",
    "input_schema": {
        "type": "object",
        "properties": {"sku": {"type": "string", "description": "The SKU code, e.g. SKU-101"}},
        "required": ["sku"],
    },
}]


# ---- the two calls -------------------------------------------------------
# Everything below is called, not run, when Python reads this file, so the
# helpers these functions use may be defined further down.
def two_model_calls(question):
    messages = [{"role": "user", "content": question}]

    # call 1: the model asks for a tool, then halts
    reply = call_model("call 1", messages)

    calls = collect_tool_calls(reply)
    if not calls:
        raise SystemExit("Call 1 answered without asking for the tool: the exit condition came early.")

    # the model's turn goes into the history exactly as it came back...
    messages.append({"role": "assistant", "content": reply.content})

    # ...then YOUR code runs each tool, and every result goes up in one user message
    results = []
    for call in calls:
        output = lookup_stock(**call.input)
        results.append({"type": "tool_result", "tool_use_id": call.id, "content": str(output)})
    messages.append({"role": "user", "content": results})

    # call 2: the model has the result, and answers
    reply = call_model("call 2", messages)

    for block in reply.content:
        if block.type == "text":
            print(block.text)


def call_model(title, messages):
    """One request and one reply: the messages sent up, and the one that comes back."""
    print(f"=== {title} ===")
    print_messages(messages)
    reply = client.messages.create(model=MODEL, max_tokens=1000, tools=TOOLS, messages=messages)
    print(f"{'- ' * 6}-+{'- ' * 29}".rstrip())
    print_msg_blocks(f"<- {reply.role}", reply.content)
    print(f"{'':<12} | stop_reason={reply.stop_reason}")
    print()
    return reply


def collect_tool_calls(reply):
    """The tool_use blocks in a reply, in the order they arrived."""
    calls = []
    for block in reply.content:
        if block.type == "tool_use":
            calls.append(block)
    return calls


# ---- one row per block ---------------------------------------------------
def print_messages(messages):
    """The list being sent, one row per block."""
    print(f"   {'role':<9} | blocks")
    print(f"{'-' * 12}-+-{'-' * 58}")
    for message in messages:
        print_msg_blocks(f"-> {message['role']}", message["content"])


def print_msg_blocks(label, content):
    """One message: its label on the first row, then one row per block."""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    first = True
    for block in content:
        print_block(label if first else "", block)
        first = False


def print_block(label, block):
    """One row: the label column, then the block's two or three main fields."""
    if not isinstance(block, dict):
        block = block.model_dump()
    kind = block["type"]
    if kind == "text":
        text = block["text"].replace("\n", " ")
        if len(text) > 52:
            text = text[:51] + "..."
        fields = f'text "{text}"'
    elif kind == "tool_use":
        fields = f'tool_use {block["name"]} #{block["id"][6:14]}'
    elif kind == "tool_result":
        fields = f'tool_result "{block["content"]}" #{block["tool_use_id"][6:14]}'
    else:
        fields = kind
    print(f"{label:<12} | {fields}")


if __name__ == "__main__":
    two_model_calls("How many units of SKU-102 do we have?")

# stop_reason was "tool_use", then "end_turn". That one field is the
# exit condition of the loop you write in the next file.
