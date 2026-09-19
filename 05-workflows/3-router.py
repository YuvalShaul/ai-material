"""Routing: a cheap classifier, then a branch you wrote.

The model still decides something — which of three branches this ticket
belongs to — but it decides it once, into a fixed vocabulary, and the
graph owns what happens next. That is the whole difference between a
router and an agent.

Structured output is the mechanism. An edge cannot be drawn from prose:
the classifier has to return one of a known set of values, and
`with_structured_output` is what makes it.

Every ticket is answered twice: once through the router, once by handing
the whole job to the big model with every instruction in one prompt. The
table at the end is the reason routers are in production.

    python 3-router.py
"""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langchain_anthropic import ChatAnthropic

SMALL, LARGE = "claude-haiku-4-5", "claude-opus-5"
PRICES = {SMALL: (1.00, 5.00), LARGE: (5.00, 25.00)}  # $ per million tokens

TICKETS = [
    "I was charged twice for order #A-8891 this month. Please sort it out.",
    "Export to CSV crashes with 'KeyError: total' on any report with a "
    "filtered column. It worked last week. Chrome 141, macOS.",
    "How do I invite a teammate to my workspace, and does it cost extra?",
]

BRANCH_PROMPTS = {
    "billing": ("You answer billing tickets. Never promise a refund; say it "
                "is escalated to billing with the order number."),
    "bug": ("You triage bug reports. Ask for the missing reproduction "
            "detail, state the likely cause, and give a workaround."),
    "how-to": ("You answer product how-to questions in three sentences, "
               "with the exact menu path."),
}


class Route(BaseModel):
    """Where a ticket goes, and how loudly."""
    category: Literal["billing", "bug", "how-to"] = Field(
        description="which queue this ticket belongs in")
    urgency: int = Field(description="1 calm, 5 furious")


class State(TypedDict):
    ticket: str
    route: Route | None
    answer: str
    calls: list[tuple[str, dict]]     # (model, usage) per call, overwritten


def price(model_name, usage) -> float:
    pin, pout = PRICES[model_name]
    return (usage["input_tokens"] * pin + usage["output_tokens"] * pout) / 1e6


# ---- the router: one small call into a fixed vocabulary ------------------
def classify(state: State) -> dict:
    model = ChatAnthropic(model=SMALL, max_tokens=300)
    out = model.with_structured_output(Route, include_raw=True).invoke(
        "Classify this support ticket.\n\n" + state["ticket"])
    return {"route": out["parsed"],
            "calls": state["calls"] + [(SMALL, out["raw"].usage_metadata)]}


def pick(state: State) -> str:
    return state["route"].category


# ---- the branches: each its own prompt, each its own model tier ----------
def answer_with(model_name: str, category: str):
    def node(state: State) -> dict:
        kwargs = {"output_config": {"effort": "low"}} if model_name == LARGE else {}
        model = ChatAnthropic(model=model_name, max_tokens=2000, **kwargs)
        reply = model.invoke(BRANCH_PROMPTS[category] + "\n\n" + state["ticket"])
        return {"answer": reply.text,
                "calls": state["calls"] + [(model_name, reply.usage_metadata)]}
    return node


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("billing", answer_with(SMALL, "billing"))
builder.add_node("bug", answer_with(LARGE, "bug"))        # the one that earns it
builder.add_node("how-to", answer_with(SMALL, "how-to"))
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", pick,
                              {"billing": "billing", "bug": "bug",
                               "how-to": "how-to"})
for branch in ("billing", "bug", "how-to"):
    builder.add_edge(branch, END)
graph = builder.compile()


# ---- the baseline: one big prompt on the big model ----------------------
ONE_BIG_PROMPT = ("You are a support agent. Decide whether this is a "
                  "billing, bug or how-to ticket, then answer it "
                  "accordingly.\n\n"
                  + "\n".join(f"- {k}: {v}" for k, v in BRANCH_PROMPTS.items()))


def baseline(ticket: str) -> tuple[str, float]:
    model = ChatAnthropic(model=LARGE, max_tokens=2000,
                          output_config={"effort": "low"})
    reply = model.invoke(ONE_BIG_PROMPT + "\n\nTicket:\n" + ticket)
    return reply.text, price(LARGE, reply.usage_metadata)


if __name__ == "__main__":
    routed_total = big_total = 0.0
    rows = []
    for ticket in TICKETS:
        result = graph.invoke({"ticket": ticket, "route": None, "answer": "",
                               "calls": []})
        routed = sum(price(m, u) for m, u in result["calls"])
        _, big = baseline(ticket)
        routed_total += routed
        big_total += big
        r = result["route"]
        rows.append((r.category, r.urgency, routed, big))
        print(f"--- {r.category} (urgency {r.urgency}) ---")
        print(result["answer"][:300].strip())
        print()

    print(f"{'route':<9} {'urgency':>7} {'routed':>9} {'one big prompt':>15}")
    for category, urgency, routed, big in rows:
        print(f"{category:<9} {urgency:>7} {routed:>9.4f} {big:>15.4f}")
    print(f"{'total':<9} {'':>7} {routed_total:>9.4f} {big_total:>15.4f}")
    print(f"\nThe router answered the same three tickets for "
          f"{routed_total / big_total:.0%} of the cost.")
    print("Two of the three never touched the expensive model.")
    print()
    print(graph.get_graph().draw_mermaid(with_styles=False))
