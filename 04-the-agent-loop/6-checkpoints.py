"""One line changes: compile(checkpointer=...). Watch what it buys.

The graph is 5-graph.py's. With a checkpointer, the runtime saves the
state after EVERY node, filed under a thread id. That has two visible
effects, and this file shows both:

  1. the history: one checkpoint per node boundary, in order
  2. continuation: invoking the same thread again starts from the saved
     state, so the conversation remembers its earlier turns

    python 6-checkpoints.py
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_anthropic import ChatAnthropic

MODEL = "claude-opus-5"
STOCK = {"SKU-101": 12, "SKU-102": 0, "SKU-103": 4}


@tool
def lookup_stock(sku: str) -> int:
    """How many units of one SKU are in the warehouse right now."""
    return STOCK.get(sku, 0)


TOOLS = [lookup_stock]
DISPATCH = {t.name: t for t in TOOLS}
model = ChatAnthropic(model=MODEL, max_tokens=1000).bind_tools(TOOLS)


def call_model(state: MessagesState) -> dict:
    reply = model.invoke(state["messages"])
    u = reply.usage_metadata
    print(f"model: sent {u['input_tokens']:>5} tokens, got {u['output_tokens']:>4}")
    return {"messages": [reply]}


def run_tools(state: MessagesState) -> dict:
    results = []
    for call in state["messages"][-1].tool_calls:
        result = DISPATCH[call["name"]].invoke(call)
        print(f"tools: {call['name']}({call['args']}) -> {result.content}")
        results.append(result)
    return {"messages": results}


def route(state: MessagesState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", run_tools)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", route, {"tools": "tools", END: END})
builder.add_edge("tools", "model")

graph = builder.compile(checkpointer=InMemorySaver())        # <- the one change
config = {"configurable": {"thread_id": "inventory-1"}}      # which saved state to use


if __name__ == "__main__":
    result = graph.invoke({"messages": [HumanMessage("How many units of SKU-102 do we have?")]}, config)
    print(result["messages"][-1].text)

    # 1. The history: the runtime wrote a checkpoint after every node.
    print()
    print("--- checkpoints for this thread, oldest first ---")
    for snap in reversed(list(graph.get_state_history(config))):
        step = snap.metadata["step"]
        print(f"step {step:>2}: {len(snap.values.get('messages', [])):>2} messages saved, "
              f"next node: {snap.next or '(finished)'}")

    # 2. Continuation: same thread, a question that only makes sense with
    #    the earlier turn in the state. Nothing is re-sent by you.
    print()
    print("--- same thread, second question ---")
    result = graph.invoke({"messages": [HumanMessage("And SKU-103?")]}, config)
    print(result["messages"][-1].text)
    print(f"({len(result['messages'])} messages in the thread now)")
