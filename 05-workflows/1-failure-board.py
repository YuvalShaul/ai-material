"""Four ways the chapter 4 agent fails, caused on purpose.

Same shape as 04-the-agent-loop/5-graph.py: model node, tools node, one
conditional edge, and nobody above the model deciding anything. Each mode
below causes one failure and prints the evidence.

    python 1-failure-board.py spin       # it stops when it feels like it
    python 1-failure-board.py spin 4     # ... and the ceiling that stops it
    python 1-failure-board.py compound   # a wrong step is never revisited
    python 1-failure-board.py overflow   # the window fills with tool output
    python 1-failure-board.py cost       # the bill grows faster than the chat

One root cause under all four: the model decides every step, and nothing
bounds it.
"""

import sys

from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_anthropic import ChatAnthropic

MODEL = "claude-opus-5"
PRICE_IN, PRICE_OUT = 5.00, 25.00  # dollars per million tokens, claude-opus-5


def dollars(usage) -> float:
    return (usage["input_tokens"] * PRICE_IN
            + usage["output_tokens"] * PRICE_OUT) / 1_000_000


# ---- the tools each mode needs -------------------------------------------
_LEAD = {"n": 4100}


@tool
def search_orders(query: str) -> str:
    """Search the order database for an order matching a free-text query."""
    # Never a match, and always one more lead. Real search tools do this:
    # fuzzy results that each suggest the next query.
    _LEAD["n"] += 1
    return (f"no exact match for {query!r}. Closest: order #{_LEAD['n']}, "
            f"a different customer, same week. Nearby orders "
            f"#{_LEAD['n'] + 1} and #{_LEAD['n'] + 2} are unchecked.")


@tool
def fetch_log(service: str) -> str:
    """Return the last 200 lines of one service's log."""
    return ("WARN pool exhausted, retrying\n" * 120) + f"# end of {service} log"


def build(tools, system=None):
    """The chapter 4 agent, unchanged except for which tools it holds."""
    model = ChatAnthropic(model=MODEL, max_tokens=1000).bind_tools(tools)
    prefix = [SystemMessage(system)] if system else []
    dispatch = {t.name: t for t in tools}
    spent = []

    def call_model(state: MessagesState) -> dict:
        reply = model.invoke(prefix + state["messages"])
        u = reply.usage_metadata
        spent.append(u)
        print(f"  call {len(spent)}: sent {u['input_tokens']:>6} tok, "
              f"got {u['output_tokens']:>4} tok, ${dollars(u):.4f}")
        return {"messages": [reply]}

    def run_tools(state: MessagesState) -> dict:
        out = []
        for call in state["messages"][-1].tool_calls:
            result = dispatch[call["name"]].invoke(call)
            print(f"    tool: {call['name']}({call['args']}) -> "
                  f"{result.content[:48]}...")
            out.append(result)
        return {"messages": out}

    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", run_tools)
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model",
        lambda s: "tools" if s["messages"][-1].tool_calls else END,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "model")
    return builder.compile(), spent


# ---- 1. non-termination ---------------------------------------------------
def spin(limit: int = 10):
    """A tool that never satisfies. The loop has no reason to stop."""
    # One ordinary line of encouragement, the kind most production system
    # prompts carry. Nothing in the graph disagrees with it.
    graph, spent = build([search_orders],
                         system="You are thorough. Never give up on a search: "
                                "every result suggests another query. Keep "
                                "working until you have found the order.")
    question = ("Find the order for the customer who called about a broken "
                "lamp last Tuesday. The order is definitely in there.")
    try:
        graph.invoke({"messages": [HumanMessage(question)]},
                     {"recursion_limit": limit})
        print(f"\nIt stopped after {len(spent)} calls -- because the model "
              f"chose to.")
        print("Nothing in your design decided that number, so nothing "
              "guarantees it next time.")
    except GraphRecursionError:
        print(f"\nGraphRecursionError after {len(spent)} model calls, "
              f"${sum(dollars(u) for u in spent):.3f} spent.")
        print("The ceiling that stopped it is the runtime's recursion_limit, "
              "a blunt number")
        print("that knows nothing about the task. Your design has no number "
              "in it at all.")


# ---- 2. compounding errors -----------------------------------------------
def compound():
    """One wrong number, already in the history. Nothing goes back to check."""
    graph, _ = build([search_orders])
    wrong = [
        HumanMessage("How many units of SKU-102 do we have?"),
        AIMessage(content="", tool_calls=[
            {"name": "search_orders", "args": {"query": "SKU-102 stock"},
             "id": "call_1"}]),
        ToolMessage(content="SKU-102: 500 units in stock", tool_call_id="call_1"),
        AIMessage("You have 500 units of SKU-102."),
        HumanMessage("Each unit sells for $12. What is that stock worth, "
                     "and can we fill an order for 400?"),
    ]
    print("  seeded history: one tool result says 500 units. The real "
          "number is 0.")
    result = graph.invoke({"messages": wrong})
    print("\n" + result["messages"][-1].text)
    print("\nThe error was in step 1 and every later step built on it.")
    print("A loop only ever looks at the last message.")


# ---- 3. context overflow --------------------------------------------------
def overflow():
    """Tool output stays in the history, and the history is re-sent."""
    graph, spent = build([fetch_log])
    question = ("Check the logs of the api, worker and cache services, one "
                "tool call at a time, and tell me what is wrong.")
    graph.invoke({"messages": [HumanMessage(question)]}, {"recursion_limit": 12})
    print()
    for i, u in enumerate(spent, 1):
        print(f"  call {i}: {u['input_tokens']:>6} input tokens")
    if len(spent) > 1:
        growth = spent[-1]["input_tokens"] / spent[0]["input_tokens"]
        print(f"\nThe last call sent {growth:.1f}x what the first one did.")
    print("Nothing dropped a log after reading it. Nothing summarised one.")


# ---- 4. cost blowup -------------------------------------------------------
def cost():
    """Every turn re-sends the whole conversation. The bill is a triangle."""
    graph, spent = build([])
    messages = [HumanMessage("In one sentence: what is a vector database?")]
    follow_ups = [
        "And in one sentence: when is it the wrong choice?",
        "One sentence: what does a hybrid search add?",
        "One sentence: what does that cost at scale?",
        "One sentence: what would you use instead for 10k documents?",
    ]
    total = 0.0
    for turn, question in enumerate([messages[0].content] + follow_ups, 1):
        messages = (messages if turn == 1
                    else messages + [HumanMessage(question)])
        result = graph.invoke({"messages": messages})
        messages = result["messages"]
        total += dollars(spent[-1])
        print(f"  turn {turn}: ${dollars(spent[-1]):.4f} this turn, "
              f"${total:.4f} so far")
    first, last = dollars(spent[0]), dollars(spent[-1])
    print(f"\nTurn 5 cost {last / first:.1f}x turn 1, for the same size of "
          f"question.")
    print("The conversation is the input, and it only grows.")


MODES = {"spin": spin, "compound": compound, "overflow": overflow, "cost": cost}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in MODES:
        sys.exit(f"usage: python 1-failure-board.py {'|'.join(MODES)}")
    print(f"--- {mode} ---")
    if mode == "spin" and len(sys.argv) > 2:
        MODES[mode](int(sys.argv[2]))   # a lower ceiling, to see it fire
    else:
        MODES[mode]()
