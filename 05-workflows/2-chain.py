"""Prompt chaining: three fixed steps and a gate between them.

No model decides what happens next here. The edges are written down, so
the path is the same on every ticket: extract, gate, draft, shorten.

Two things to watch:

  * the state is not a message list. It is a small record with a key per
    step, plus a `log` key whose reducer appends, so every node can add a
    line without reading what the others wrote.
  * every step runs on the cheapest model that can do it, because a step
    you wrote is a step you can size.

    python 2-chain.py
"""

from operator import add
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langchain_anthropic import ChatAnthropic

SMALL, LARGE = "claude-haiku-4-5", "claude-opus-5"
PRICES = {SMALL: (1.00, 5.00), LARGE: (5.00, 25.00)}  # $ per million tokens

TICKET = """Subject: still charged after cancelling

I cancelled my Pro plan on the 3rd (order #A-8891) but I was charged
again on the 5th. I have the email confirming the cancellation. Can you
refund the second charge? This is the second time this has happened.
"""


class Extracted(BaseModel):
    """The fields a support reply needs before it can be written."""
    order_number: str = Field(description="order number, or '' if absent")
    category: Literal["billing", "bug", "how-to"]
    customer_is_angry: bool


class State(TypedDict):
    ticket: str
    fields: Extracted | None
    draft: str
    reply: str
    log: Annotated[list[str], add]       # the reducer: every node appends


def price(model_name, usage) -> float:
    pin, pout = PRICES[model_name]
    return (usage["input_tokens"] * pin + usage["output_tokens"] * pout) / 1e6


def note(model_name, step, usage) -> str:
    return (f"{step:<9} {model_name:<18} {usage['input_tokens']:>5} in "
            f"{usage['output_tokens']:>5} out  ${price(model_name, usage):.4f}")


# ---- step 1: extract, on the small model ---------------------------------
def extract(state: State) -> dict:
    model = ChatAnthropic(model=SMALL, max_tokens=500)
    structured = model.with_structured_output(Extracted, include_raw=True)
    out = structured.invoke(
        "Extract the fields from this support ticket.\n\n" + state["ticket"])
    usage = out["raw"].usage_metadata
    return {"fields": out["parsed"], "log": [note(SMALL, "extract", usage)]}


# ---- the gate: plain Python, no model ------------------------------------
def gate(state: State) -> str:
    """The cheapest check in the pipeline, and the one that saves the most."""
    if not state["fields"].order_number:
        return "hand_over"
    return "draft"


def hand_over(state: State) -> dict:
    return {"reply": "No order number in the ticket — queued for a human.",
            "log": ["gate      no order number, stopped before the model"]}


# ---- step 2: draft, on the small model -----------------------------------
def draft(state: State) -> dict:
    model = ChatAnthropic(model=SMALL, max_tokens=700)
    f = state["fields"]
    reply = model.invoke(
        f"Write a support reply to this {f.category} ticket. Order "
        f"{f.order_number}. Do not promise a refund; say it has been "
        f"escalated.\n\n{state['ticket']}")
    return {"draft": reply.text, "log": [note(SMALL, "draft", reply.usage_metadata)]}


# ---- step 3: shorten, on the large model ---------------------------------
def shorten(state: State) -> dict:
    # effort "low": this is a rewrite, not a problem to think about. On
    # claude-opus-5 thinking is on by default and its tokens count towards
    # max_tokens, so a small cap plus high effort truncates the answer.
    model = ChatAnthropic(model=LARGE, max_tokens=2000,
                          output_config={"effort": "low"})
    reply = model.invoke(
        "Rewrite this support reply in at most 60 words. Keep the order "
        "number and the escalation. Plain sentences, no bullet "
        f"points.\n\n{state['draft']}")
    return {"reply": reply.text, "log": [note(LARGE, "shorten", reply.usage_metadata)]}


builder = StateGraph(State)
builder.add_node("extract", extract)
builder.add_node("draft", draft)
builder.add_node("shorten", shorten)
builder.add_node("hand_over", hand_over)
builder.add_edge(START, "extract")
builder.add_conditional_edges("extract", gate,
                              {"draft": "draft", "hand_over": "hand_over"})
builder.add_edge("draft", "shorten")
builder.add_edge("shorten", END)
builder.add_edge("hand_over", END)
graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({"ticket": TICKET, "fields": None, "draft": "",
                           "reply": "", "log": []})
    print("extracted:", result["fields"], "\n")
    for line in result["log"]:
        print(" ", line)
    total = sum(float(l.rsplit("$", 1)[1]) for l in result["log"] if "$" in l)
    print(f"  {'total':<9} {'':<18} {'':>5}    {'':>5}       ${total:.4f}\n")
    print(result["reply"])
    print()
    print(graph.get_graph().draw_mermaid(with_styles=False))
