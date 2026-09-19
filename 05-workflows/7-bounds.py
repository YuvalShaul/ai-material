"""Restraint: the same bounds, placed one layer up.

Sessions 1 to 3 took decisions away from the model by drawing them as
nodes and edges. Sometimes you cannot: the paths are not enumerable, and
the agent has to keep deciding. Then you keep the delegation and bound it.

`create_agent` is `langchain`, one layer above `langgraph`. It builds the
model/tools graph for you, and middleware is where your bounds go — each
one a hook the loop calls, not a node you wire:

  * ModelCallLimitMiddleware   a hard ceiling on calls, in the design
  * HumanInTheLoopMiddleware   an interrupt before a consequential tool
  * @wrap_tool_call            a guardrail every tool call passes through

The checkpointer is a file, so the approval can come after this process
has exited. That is chapter 4's mechanism, unchanged.

    python 7-bounds.py graph            # what create_agent built
    python 7-bounds.py run              # runs until it needs a human
    python 7-bounds.py resume approve   # ... and finishes it
    python 7-bounds.py resume reject
    python 7-bounds.py run 249          # over the ceiling: approved, then refused
"""

import sys
from pathlib import Path

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langchain.agents import create_agent
from langchain.agents.middleware import (HumanInTheLoopMiddleware,
                                         ModelCallLimitMiddleware,
                                         wrap_tool_call)
from langchain_anthropic import ChatAnthropic

MODEL = "claude-opus-5"
DB = Path(__file__).with_name("bounds.sqlite")
def thread(amount) -> dict:
    """One thread per amount, so the two runs of the lab do not collide."""
    return {"configurable": {"thread_id": f"refund-{amount:g}"}}
REFUND_CEILING = 100.0

ORDERS = {
    "A-8891": {"customer": "R. Dale", "plan": "Pro monthly",
               "charges": ["2026-09-03  $80.00  paid",
                           "2026-09-05  $80.00  paid (duplicate)"]},
    # The same mistake, an order of magnitude up: under the gate, over the
    # ceiling. This is the one the guardrail refuses.
    "B-2210": {"customer": "M. Okonjo", "plan": "Team annual",
               "charges": ["2026-09-03  $249.00  paid",
                           "2026-09-05  $249.00  paid (duplicate)"]},
}
ORDER_FOR = {80.0: "A-8891", 249.0: "B-2210"}


@tool
def lookup_order(order_id: str) -> str:
    """Look up one order by its id."""
    order = ORDERS.get(order_id)
    return str(order) if order else f"no order {order_id}"


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund money to the customer. This moves real money."""
    return f"refunded ${amount:.2f} on {order_id}"


# ---- the guardrail: every tool call passes through it ---------------------
@wrap_tool_call
def refund_ceiling(request, handler):
    """A rule in code, not in the prompt. The model cannot argue with it."""
    call = request.tool_call
    if call["name"] == "issue_refund" and call["args"].get("amount", 0) > REFUND_CEILING:
        return ToolMessage(
            content=(f"refused: ${call['args']['amount']:.2f} is over the "
                     f"${REFUND_CEILING:.2f} limit. A manager must do this one."),
            tool_call_id=call["id"], status="error")
    return handler(request)


def build(checkpointer):
    return create_agent(
        model=ChatAnthropic(model=MODEL, max_tokens=2000,
                            output_config={"effort": "low"}),
        tools=[lookup_order, issue_refund],
        system_prompt=("You are a support agent. Refund customers who were "
                       "charged in error. Look the order up first."),
        middleware=[
            ModelCallLimitMiddleware(thread_limit=6, exit_behavior="end"),
            HumanInTheLoopMiddleware(interrupt_on={"issue_refund": True}),
            refund_ceiling,
        ],
        checkpointer=checkpointer,
    )


def task(amount) -> str:
    order = ORDER_FOR.get(amount, "A-8891")
    return (f"The customer on order {order} was charged twice. Refund the "
            f"duplicate charge of ${amount:g}.")


def show(messages):
    for m in messages:
        kind = type(m).__name__
        if getattr(m, "tool_calls", None):
            for c in m.tool_calls:
                print(f"  {kind:<12} -> {c['name']}({c['args']})")
        elif m.text:
            print(f"  {kind:<12} {m.text[:96]}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""

    with SqliteSaver.from_conn_string(str(DB)) as saver:
        agent = build(saver)

        if mode == "graph":
            print(agent.get_graph().draw_mermaid(with_styles=False))

        elif mode == "run":
            amount = float(sys.argv[2]) if len(sys.argv) > 2 else 80
            out = agent.invoke(
                {"messages": [{"role": "user", "content": task(amount)}]},
                thread(amount))
            show(out["messages"])
            if "__interrupt__" in out:
                request = out["__interrupt__"][0].value["action_requests"][0]
                print(f"\nPAUSED before {request['name']}({request['args']})")
                print("The state is on disk. Kill this process; the run "
                      "survives it.")
                print(f"  python 7-bounds.py resume approve {amount:g}")
            else:
                print("\nfinished without asking anyone")

        elif mode == "resume":
            decision = sys.argv[2] if len(sys.argv) > 2 else "approve"
            amount = float(sys.argv[3]) if len(sys.argv) > 3 else 80
            payload = ({"type": "approve"} if decision == "approve"
                       else {"type": "reject",
                             "message": "Refunds need a manager this week."})
            out = agent.invoke(Command(resume={"decisions": [payload]}),
                               thread(amount))
            show(out["messages"][-4:])
            print(f"\n{out['messages'][-1].text[:300]}")

        else:
            sys.exit("usage: python 7-bounds.py graph | run [amount] | "
                     "resume approve|reject [amount]")
