"""The whole chapter in one graph.

    route -> fan out over sources -> aggregate -> evaluate -> write

Every shape from sessions 1 and 2, joined up, with the two things a
workflow is for on show: a model tier chosen per node, and a cost printed
per node.

The sources are a dict. Chapter 6 makes them a real retrieval step; the
shape of the graph does not change when it does.

    python 6-pipeline.py            # the happy path
    python 6-pipeline.py --fail     # one source dies mid-run

The second one is the point of a pipeline: the branch that failed is the
branch that failed, and you can say so. In a level 4 agent the same
outage reads as "turn 14 went wrong".
"""

import sys
from operator import add
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langchain_anthropic import ChatAnthropic

SMALL, LARGE = "claude-haiku-4-5", "claude-opus-5"
PRICES = {SMALL: (1.00, 5.00), LARGE: (5.00, 25.00)}
MAX_REVISIONS = 2
INJECT_FAILURE = "--fail" in sys.argv

REQUEST = ("Write the briefing for tomorrow's incident review of the CSV "
           "export outage.")

SOURCES = {
    "alerts": ("14:02 export_job error rate 100%. 14:05 queue depth normal. "
               "18:10 error rate back to 0 after rollback of schema change."),
    "postmortem": ("Cause: migration dropped the total column while the "
                   "export job still selected it. The job swallowed the "
                   "exception, so detection took four hours."),
    "tickets": ("Nine customers reported failed exports between 14:10 and "
                "17:50. Two asked for the data by email instead. One "
                "threatened to cancel."),
}


class Kind(BaseModel):
    kind: Literal["briefing", "faq"] = Field(
        description="briefing needs research; faq is answered from memory")


class Judgement(BaseModel):
    ok: bool = Field(description="true if the draft is ready to send")
    problem: str = Field(description="what to fix, empty when ok")


class State(TypedDict):
    request: str
    kind: str
    notes: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    draft: str
    verdict: str
    revisions: int
    final: str
    spend: Annotated[list[tuple[str, str, float]], add]   # node, model, $


def price(model_name, usage) -> float:
    pin, pout = PRICES[model_name]
    return (usage["input_tokens"] * pin + usage["output_tokens"] * pout) / 1e6


def model_for(name: str, tokens: int = 2000):
    kwargs = {"output_config": {"effort": "low"}} if name == LARGE else {}
    return ChatAnthropic(model=name, max_tokens=tokens, **kwargs)


# ---- 1. route: one small call, fixed vocabulary --------------------------
def route(state: State) -> dict:
    out = model_for(SMALL, 300).with_structured_output(
        Kind, include_raw=True).invoke("Classify this request.\n\n"
                                       + state["request"])
    return {"kind": out["parsed"].kind,
            "spend": [("route", SMALL, price(SMALL, out["raw"].usage_metadata))]}


def after_route(state: State):
    if state["kind"] == "faq":
        return "answer_directly"
    return [Send("research", {"name": n, "text": t}) for n, t in SOURCES.items()]


# ---- 2. fan out: one branch per source, in parallel ----------------------
def research(payload: dict) -> dict:
    if INJECT_FAILURE and payload["name"] == "tickets":
        # Containment: the branch fails, the pipeline does not.
        return {"errors": [f"{payload['name']}: source unavailable"]}
    reply = model_for(SMALL, 400).invoke(
        "Two bullet points, facts only, for an incident review:\n\n"
        + payload["text"])
    return {"notes": [f"[{payload['name']}]\n{reply.text}"],
            "spend": [("research:" + payload["name"], SMALL,
                       price(SMALL, reply.usage_metadata))]}


# ---- 3. aggregate: the expensive node, once ------------------------------
def aggregate(state: State) -> dict:
    missing = ("\n\nSources unavailable: " + ", ".join(state["errors"])
               if state["errors"] else "")
    reply = model_for(LARGE).invoke(
        "Write the incident review briefing: what happened, why it took so "
        "long to notice, what we are changing. Name any missing source in "
        f"one line at the end.{missing}\n\n" + "\n\n".join(state["notes"]))
    return {"draft": reply.text,
            "spend": [("aggregate", LARGE, price(LARGE, reply.usage_metadata))]}


# ---- 4. evaluate: a cheap judge, and a counter ---------------------------
def evaluate(state: State) -> dict:
    out = model_for(SMALL, 500).with_structured_output(
        Judgement, include_raw=True).invoke(
        "Is this briefing ready for an incident review? It must state the "
        "cause, the detection delay and at least one action, and must name "
        "any missing source.\n\n" + state["draft"])
    verdict = out["parsed"]
    return {"verdict": "" if verdict.ok else verdict.problem,
            "revisions": state["revisions"] + 1,
            "spend": [("evaluate", SMALL,
                       price(SMALL, out["raw"].usage_metadata))]}


def after_evaluate(state: State) -> str:
    if not state["verdict"]:
        return "write"
    if state["revisions"] >= MAX_REVISIONS:
        return "write"              # the cap: ship the best draft, flagged
    return "aggregate"


# ---- 5. write: cheap again -----------------------------------------------
def write(state: State) -> dict:
    reply = model_for(SMALL, 1500).invoke(
        "Format this as a briefing with a one-line summary followed by "
        "three short sections. Keep every fact.\n\n" + state["draft"])
    return {"final": reply.text,
            "spend": [("write", SMALL, price(SMALL, reply.usage_metadata))]}


def answer_directly(state: State) -> dict:
    reply = model_for(SMALL, 800).invoke(state["request"])
    return {"final": reply.text,
            "spend": [("answer_directly", SMALL,
                       price(SMALL, reply.usage_metadata))]}


builder = StateGraph(State)
for name, fn in [("route", route), ("research", research),
                 ("aggregate", aggregate), ("evaluate", evaluate),
                 ("write", write), ("answer_directly", answer_directly)]:
    builder.add_node(name, fn)
builder.add_edge(START, "route")
builder.add_conditional_edges("route", after_route,
                              ["research", "answer_directly"])
builder.add_edge("research", "aggregate")
builder.add_edge("aggregate", "evaluate")
builder.add_conditional_edges("evaluate", after_evaluate,
                              {"aggregate": "aggregate", "write": "write"})
builder.add_edge("write", END)
builder.add_edge("answer_directly", END)
graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {"request": REQUEST, "kind": "", "notes": [], "errors": [],
         "draft": "", "verdict": "", "revisions": 0, "final": "", "spend": []},
        {"recursion_limit": 25})

    print(f"route: {result['kind']}, revisions: {result['revisions']}")
    if result["errors"]:
        print(f"contained: {', '.join(result['errors'])} — the other "
              f"{len(result['notes'])} branches finished")
    print()
    for node, model_name, cost in result["spend"]:
        print(f"  {node:<20} {model_name:<18} ${cost:.4f}")
    print(f"  {'total':<20} {'':<18} ${sum(c for _, _, c in result['spend']):.4f}")
    print()
    print(result["final"])
    print()
    print(graph.get_graph().draw_mermaid(with_styles=False))
