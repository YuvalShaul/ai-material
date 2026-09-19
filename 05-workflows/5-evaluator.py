"""Evaluator and optimiser: a cycle you drew, with a number on it.

The model writes a function; a test run judges it; the failure goes back
as the next prompt. That is a loop again — but it is your loop: the edge
is in the graph, the judge is a subprocess, and the counter in the state
decides when to stop trying.

An unbounded feedback edge is an outage waiting for a bad day, so
MAX_ATTEMPTS is the first guardrail in this course.

The generated code runs in a subprocess with a timeout. Running code a
model wrote is itself a decision — session 4 comes back to it.

    python 5-evaluator.py
"""

import subprocess
import sys
import textwrap
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_anthropic import ChatAnthropic

MODEL = "claude-opus-5"
MAX_ATTEMPTS = 3

TASK = """Write a Python function:

    def parse_duration(text: str) -> int

It converts a duration string into whole seconds. Handle hours, minutes
and seconds in any combination, for example '90s', '2h', '1h30m'.
Return only the function, no explanation, no tests, no markdown fence.
"""

# The judge. The model is never shown these; it only ever sees which one
# failed, which is what makes this a feedback loop rather than a hint.
TESTS = """
assert parse_duration("90s") == 90
assert parse_duration("2h") == 7200
assert parse_duration("1h30m") == 5400
assert parse_duration("1h 30m") == 5400      # a space between the parts
assert parse_duration("") == 0               # empty string is zero
assert parse_duration("1h30m15s") == 5415
"""


class State(TypedDict):
    code: str
    feedback: str
    attempts: int
    passed: bool


def generate(state: State) -> dict:
    model = ChatAnthropic(model=MODEL, max_tokens=4000,
                          output_config={"effort": "low"})
    prompt = TASK if not state["feedback"] else (
        f"{TASK}\nYour last attempt failed this check:\n"
        f"{state['feedback']}\n\nHere it is:\n{state['code']}\n\nFix it.")
    reply = model.invoke(prompt)
    code = reply.text.strip().removeprefix("```python").removesuffix("```")
    return {"code": code.strip(), "attempts": state["attempts"] + 1}


def run_tests(state: State) -> dict:
    """The evaluator. No model in it: a real run, or nothing."""
    program = state["code"] + "\n" + TESTS + "\nprint('all tests passed')\n"
    try:
        done = subprocess.run([sys.executable, "-c", program],
                              capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return {"passed": False, "feedback": "the function did not finish"}
    if done.returncode == 0:
        return {"passed": True, "feedback": ""}
    last = done.stderr.strip().splitlines()[-1]
    failing = [l.strip() for l in done.stderr.splitlines()
               if "assert parse_duration" in l]
    return {"passed": False,
            "feedback": (failing[-1] if failing else last) + f"  ({last})"}


def again(state: State) -> str:
    if state["passed"]:
        return END
    if state["attempts"] >= MAX_ATTEMPTS:
        return END                       # the cap, and the reason for it
    return "generate"


builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("test", run_tests)
builder.add_edge(START, "generate")
builder.add_edge("generate", "test")
builder.add_conditional_edges("test", again, {"generate": "generate", END: END})
graph = builder.compile()


if __name__ == "__main__":
    state = {"code": "", "feedback": "", "attempts": 0, "passed": False}
    result = graph.invoke(state, {"recursion_limit": 2 * MAX_ATTEMPTS + 2})

    print(f"attempts: {result['attempts']} of {MAX_ATTEMPTS}")
    print(f"passed:   {result['passed']}")
    if not result["passed"]:
        print(f"last failure: {result['feedback']}")
    print("\n--- the function it settled on ---")
    print(textwrap.indent(result["code"], "  "))
    print()
    print(graph.get_graph().draw_mermaid(with_styles=False))
