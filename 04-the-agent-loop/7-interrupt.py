"""interrupt() raises. Command(resume=...) replays. No model in this file.

The runtime does not care what a node does, so the mechanism is easiest
to watch with three plain functions and a counter. Three experiments:

  A. the pause: interrupt() raises, the runtime catches it, and the
     checkpoint written after the PREVIOUS node is where the run parks
  B. the resume: the interrupted node runs again from its first line,
     and this time interrupt() returns the value you passed in
  C. two things that break it: a node that catches the exception, and a
     state value that cannot be serialized

    python 7-interrupt.py
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    draft: str
    decision: str


runs = {"approve": 0}


def write(state: State) -> dict:
    return {"draft": "delete 3 files"}


def approve(state: State) -> dict:
    runs["approve"] += 1
    print(f"  approve: run {runs['approve']} — everything above interrupt() runs on every pass")
    try:
        decision = interrupt({"question": "OK to proceed?", "draft": state["draft"]})
    except GraphInterrupt as e:
        print(f"  approve: interrupt() RAISED {type(e).__name__} — re-raising, or the pause is lost")
        raise
    print(f"  approve: interrupt() RETURNED {decision!r}")
    return {"decision": decision}


def act(state: State) -> dict:
    return {"draft": f"{state['draft']} — {state['decision']}"}


builder = StateGraph(State)
builder.add_node("write", write)
builder.add_node("approve", approve)
builder.add_node("act", act)
builder.add_edge(START, "write")
builder.add_edge("write", "approve")
builder.add_edge("approve", "act")
builder.add_edge("act", END)
graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "1"}}


# --- A. the pause ---------------------------------------------------------
print("A. invoke — the run stops inside `approve`")
paused = graph.invoke({"draft": "", "decision": ""}, config)
print("  returned :", paused)
snap = graph.get_state(config)
print("  parked at:", snap.values, "next:", snap.next)
print("  checkpoints so far, oldest first:")
for s in reversed(list(graph.get_state_history(config))):
    print(f"    step {s.metadata['step']:>2}  {s.values}  next={s.next}")
print("  -> nothing is running now. The thread is a row in the checkpointer.")

# --- B. the resume --------------------------------------------------------
print()
print("B. invoke again with Command(resume=...) — `approve` runs from the top")
done = graph.invoke(Command(resume="yes"), config)
print("  returned :", done)
print(f"  approve ran {runs['approve']} times for one decision")

# --- C. two ways to break it ---------------------------------------------
print()
print("C1. a node that catches Exception swallows the pause")


def careless(state: State) -> dict:
    try:
        decision = interrupt("OK?")
    except Exception as e:                     # GraphInterrupt IS an Exception
        print(f"  careless: caught {type(e).__name__}, carrying on as if answered")
        decision = "assumed yes"
    return {"decision": decision}


b2 = StateGraph(State)
b2.add_node("careless", careless)
b2.add_edge(START, "careless")
b2.add_edge("careless", END)
g2 = b2.compile(checkpointer=InMemorySaver())
print("  returned :", g2.invoke({"draft": "", "decision": ""}, {"configurable": {"thread_id": "2"}}))
print("  -> no __interrupt__ key. The run never paused, and nobody was asked.")

print()
print("C2. state that cannot be serialized cannot be checkpointed")


class Handle:                                  # a socket, a file, a lock...
    pass


class BadState(TypedDict):
    conn: object


def open_conn(state: BadState) -> dict:
    return {"conn": Handle()}


b3 = StateGraph(BadState)
b3.add_node("open_conn", open_conn)
b3.add_edge(START, "open_conn")
b3.add_edge("open_conn", END)
g3 = b3.compile(checkpointer=InMemorySaver())
try:
    g3.invoke({"conn": None}, {"configurable": {"thread_id": "3"}})
except TypeError as e:
    print(f"  {type(e).__name__}: {e}")
print("  -> the node ran fine. The checkpoint after it is what failed.")
