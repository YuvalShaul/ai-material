"""1-protocol.py again, written in langchain-core's types.

Nothing new happens here. The same two calls, the same halt, the same
result handed back. What changes is the vocabulary: the raw blocks and
dicts of session 1 become classes that every layer above will pass
around — AIMessage, ToolMessage, and a tool built by @tool.

    python 4-core-types.py
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

MODEL = "claude-opus-5"
STOCK = {"SKU-101": 12, "SKU-102": 0, "SKU-103": 4}


@tool
def lookup_stock(sku: str) -> int:
    """How many units of one SKU are in the warehouse right now."""
    return STOCK.get(sku, 0)


# @tool built the schema you wrote by hand in session 1. Look at it.
print("name       :", lookup_stock.name)
print("description:", lookup_stock.description)
print("args       :", lookup_stock.args)
print()

# ChatAnthropic is a BaseChatModel. bind_tools attaches the schemas to
# every request — the `tools=` argument of session 1, made permanent.
model = ChatAnthropic(model=MODEL, max_tokens=1000).bind_tools([lookup_stock])

messages = [HumanMessage("How many units of SKU-102 do we have?")]

# --- call 1: the model asks for a tool, then halts ---------------------
reply = model.invoke(messages)                       # an AIMessage
print("--- call 1 ---")
print("type       :", type(reply).__name__)
print("tool_calls :", reply.tool_calls)              # parsed for you: name, args, id
print("usage      :", reply.usage_metadata)
messages.append(reply)                               # the assistant turn, as it came back

# --- you run the tool ----------------------------------------------------
for call in reply.tool_calls:
    result = lookup_stock.invoke(call)               # a ToolMessage, tool_call_id filled in
    print("you ran    :", repr(result))
    messages.append(result)

# --- call 2: the model has the result, and answers ----------------------
reply = model.invoke(messages)
print()
print("--- call 2 ---")
print("tool_calls :", reply.tool_calls)              # empty: the exit condition
print(reply.text)
