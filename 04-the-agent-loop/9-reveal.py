"""What create_agent builds, printed.

The langchain layer's create_agent takes a model and some tools and
returns a compiled langgraph graph. Print that graph, and compare it with
what 5-graph.py printed: it is the two nodes and the conditional edge
you wrote by hand.

Chapter 5 is about what else create_agent offers. This file only shows
that there is nothing hidden in it.

    python 9-reveal.py
"""

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

MODEL = "claude-opus-5"
STOCK = {"SKU-101": 12, "SKU-102": 0, "SKU-103": 4}


@tool
def lookup_stock(sku: str) -> int:
    """How many units of one SKU are in the warehouse right now."""
    return STOCK.get(sku, 0)


agent = create_agent(model=ChatAnthropic(model=MODEL, max_tokens=1000), tools=[lookup_stock])

print(type(agent).__name__)                          # CompiledStateGraph — a langgraph object
print(agent.get_graph().draw_mermaid(with_styles=False))

result = agent.invoke({"messages": [{"role": "user", "content":
    "Which of SKU-101, SKU-102 and SKU-103 need reordering? Anything under 5 units does."}]})
for m in result["messages"]:
    print(f"{type(m).__name__:<13}| {m.text[:80] if m.text else m.tool_calls}")
