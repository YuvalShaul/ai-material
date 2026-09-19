"""Fan-out and fan-in: many branches at once, one node to join them.

This is a different axis from "who decides". Nobody here chooses a path:
the graph summarises every document, always. What changes is how many run
at the same time.

`Send` is the dynamic form: the number of branches is decided at run time
from the state, not drawn in the graph. Because the branches all write to
the same key, the reducer on that key stops them overwriting each other —
this is where the reducers of the last lesson start to matter.

    python 4-fanout.py
"""

import time
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langchain_anthropic import ChatAnthropic

SMALL, LARGE = "claude-haiku-4-5", "claude-opus-5"
PRICES = {SMALL: (1.00, 5.00), LARGE: (5.00, 25.00)}

DOCS = {
    "runbook": ("Pager runbook: when the queue depth alert fires, first "
                "check the consumer count in the dashboard. If consumers "
                "are zero, restart the worker deployment. If consumers are "
                "healthy, the producer is ahead: raise the batch size and "
                "tell the data team. Never purge the queue."),
    "postmortem": ("Incident 2026-07-02: exports failed for four hours. "
                   "Cause: a schema change dropped the total column while "
                   "the export job still selected it. Detection was slow "
                   "because the job swallowed exceptions. Action: fail "
                   "loudly, add a contract test, alert on job error rate."),
    "onboarding": ("New engineer setup: request access to the staging "
                   "account, install the CLI, run the seed script, and pair "
                   "with a reviewer on your first change. Do not deploy to "
                   "production in your first week. Ask for the on-call "
                   "shadow rota in week two."),
}


class State(TypedDict):
    docs: dict[str, str]
    summaries: Annotated[list[str], add]    # every branch appends here
    timings: Annotated[list[float], add]
    digest: str


def price(model_name, usage) -> float:
    pin, pout = PRICES[model_name]
    return (usage["input_tokens"] * pin + usage["output_tokens"] * pout) / 1e6


COST = []


# ---- one branch, run once per document -----------------------------------
def summarize(payload: dict) -> dict:
    """Receives what Send handed it, not the whole state."""
    started = time.perf_counter()
    model = ChatAnthropic(model=SMALL, max_tokens=400)
    reply = model.invoke("One sentence, no preamble. Summarise this note "
                         f"for a colleague:\n\n{payload['text']}")
    took = time.perf_counter() - started
    COST.append(price(SMALL, reply.usage_metadata))
    print(f"  {payload['name']:<12} {took:>5.1f}s  {reply.text[:64]}...")
    return {"summaries": [f"{payload['name']}: {reply.text}"], "timings": [took]}


# ---- the fan-out: an edge that returns a list of Sends --------------------
def fan_out(state: State) -> list[Send]:
    return [Send("summarize", {"name": name, "text": text})
            for name, text in state["docs"].items()]


# ---- the fan-in: one node, after every branch has finished ----------------
def aggregate(state: State) -> dict:
    model = ChatAnthropic(model=LARGE, max_tokens=2000,
                          output_config={"effort": "low"})
    reply = model.invoke("Turn these note summaries into one short briefing "
                         "for a new team member. Three sentences.\n\n"
                         + "\n".join(state["summaries"]))
    COST.append(price(LARGE, reply.usage_metadata))
    return {"digest": reply.text}


builder = StateGraph(State)
builder.add_node("summarize", summarize)
builder.add_node("aggregate", aggregate)
builder.add_conditional_edges(START, fan_out, ["summarize"])
builder.add_edge("summarize", "aggregate")
builder.add_edge("aggregate", END)
graph = builder.compile()


if __name__ == "__main__":
    print(f"fanning out to {len(DOCS)} branches")
    started = time.perf_counter()
    result = graph.invoke({"docs": DOCS, "summaries": [], "timings": [],
                           "digest": ""})
    elapsed = time.perf_counter() - started

    print(f"\nwall clock      {elapsed:>5.1f}s")
    print(f"sum of branches {sum(result['timings']):>5.1f}s  "
          f"<- what one after another would have cost you")
    print(f"total spend     ${sum(COST):.4f}")
    print(f"\n{result['digest']}")
    print()
    print(graph.get_graph().draw_mermaid(with_styles=False))
