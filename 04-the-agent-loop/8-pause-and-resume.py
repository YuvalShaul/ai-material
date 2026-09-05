"""Pause before a destructive tool. Kill the process. Come back. Resume.

The graph is 5-graph.py's, with two changes:
  - the tools node calls interrupt() before it runs delete_file
  - the checkpointer is a SQLite file, so the saved state outlives the process

There is also one deliberate mistake, marked HAZARD below: a side effect
placed ABOVE the interrupt() call. Watch the audit log to see what that
costs.

    python 8-pause-and-resume.py            # runs until the agent wants to delete, then exits
    python 8-pause-and-resume.py --approve  # new process: resumes the thread, the deletes run
    python 8-pause-and-resume.py --reject   # new process: resumes, the deletes are refused
    python 8-pause-and-resume.py --reset    # forget the thread, rebuild the scratch files
"""

import pathlib
import sys

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from langchain_anthropic import ChatAnthropic

MODEL = "claude-opus-5"
HERE = pathlib.Path(__file__).parent
SCRATCH = HERE / "scratch"
AUDIT = HERE / "audit.log"
DB = HERE / "checkpoints.sqlite"
TASK = "Delete every .tmp file in the working directory, then tell me what is left."
DESTRUCTIVE = {"delete_file"}


def rebuild_scratch():
    SCRATCH.mkdir(exist_ok=True)
    for old in SCRATCH.iterdir():
        old.unlink()
    for name in ("report.txt", "notes.md", "build.tmp", "cache.tmp"):
        (SCRATCH / name).write_text(f"contents of {name}\n")


# ---- tools ----------------------------------------------------------------
@tool
def list_files() -> str:
    """List the files in the working directory, one name per line."""
    return "\n".join(sorted(p.name for p in SCRATCH.iterdir())) or "(empty)"


@tool
def delete_file(name: str) -> str:
    """Delete one file from the working directory. This cannot be undone."""
    path = SCRATCH / name
    if not path.exists():
        return f"no such file: {name}"
    path.unlink()
    return f"deleted {name}"


TOOLS = [list_files, delete_file]
DISPATCH = {t.name: t for t in TOOLS}
model = ChatAnthropic(model=MODEL, max_tokens=1000).bind_tools(TOOLS)


# ---- nodes ----------------------------------------------------------------
def call_model(state: MessagesState) -> dict:
    reply = model.invoke(state["messages"])
    u = reply.usage_metadata
    print(f"model: sent {u['input_tokens']:>5} tokens, got {u['output_tokens']:>4}")
    return {"messages": [reply]}


def run_tools(state: MessagesState) -> dict:
    calls = state["messages"][-1].tool_calls
    risky = [c for c in calls if c["name"] in DESTRUCTIVE]

    if risky:
        # HAZARD: a side effect above interrupt(). This line runs on the
        # pause AND again on the resume. Count the lines in audit.log.
        with AUDIT.open("a") as f:
            f.write(f"approval requested: {[c['args'] for c in risky]}\n")
        print(f"tools: asking for approval — {AUDIT.name} written")

        decision = interrupt({
            "about_to_run": [f"{c['name']}({c['args']})" for c in risky],
            "answer_with": "--approve or --reject",
        })
        # everything from here on runs once, on the resume only
        if decision != "approve":
            return {"messages": [
                ToolMessage(content="refused by the operator", tool_call_id=c["id"]) for c in calls
            ]}

    results = []
    for call in calls:
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

config = {"configurable": {"thread_id": "lab"}}


# ---- the process ------------------------------------------------------------
def main(argv: list[str]) -> None:
    if "--reset" in argv:
        DB.unlink(missing_ok=True)
        AUDIT.unlink(missing_ok=True)
        rebuild_scratch()
        print("thread forgotten, scratch/ rebuilt, audit.log removed")
        return
    if not SCRATCH.exists():
        rebuild_scratch()

    with SqliteSaver.from_conn_string(str(DB)) as saver:     # the state lives HERE, not in RAM
        graph = builder.compile(checkpointer=saver)
        parked = graph.get_state(config).next               # () unless a run is paused

        if "--approve" in argv or "--reject" in argv:
            if not parked:
                print("nothing is paused on this thread — run without flags first")
                return
            print(f"loaded checkpoint from {DB.name}: next node = {parked}")
            print(f"resuming — the '{parked[0]}' node runs again from its first line")
            decision = "approve" if "--approve" in argv else "reject"
            result = graph.invoke(Command(resume=decision), config)
        elif parked:
            print(f"this thread is paused at {parked}; finish it with --approve / --reject, "
                  f"or start over with --reset")
            return
        else:
            result = graph.invoke({"messages": [HumanMessage(TASK)]}, config)

    if "__interrupt__" in result:
        print()
        print("PAUSED:", result["__interrupt__"][0].value)
        print(f"The state is in {DB.name}. This process is about to exit; nothing keeps running.")
        return
    print()
    print(result["messages"][-1].text)
    print()
    print("audit.log now reads:")
    print(AUDIT.read_text().rstrip() if AUDIT.exists() else "(no approvals were requested)")


if __name__ == "__main__":
    main(sys.argv[1:])
