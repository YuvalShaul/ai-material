"""The port: 2-loop.py as a LangGraph graph.

Two nodes and one conditional edge. Every line here has a line in
2-loop.py that it replaces — the lesson's table lists the pairs.

The difference is not what runs. It is WHO calls WHOM: you no longer
call the model and the tools. You hand two functions to a runtime, and
the runtime calls them.

    python 5-graph.py
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
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


# ---- the two nodes: each gets the state, returns what it adds ------------
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


# ---- the exit condition, as an edge --------------------------------------
def route(state: MessagesState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


# ---- the loop, as edges ---------------------------------------------------
builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", run_tools)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", route, {"tools": "tools", END: END})
builder.add_edge("tools", "model")
graph = builder.compile()


if __name__ == "__main__":
    question = "Which of SKU-101, SKU-102 and SKU-103 need reordering? Anything under 5 units does."
    result = graph.invoke({"messages": [HumanMessage(question)]})
    print()
    print(result["messages"][-1].text)
    print()
    print(graph.get_graph().draw_mermaid(with_styles=False))
