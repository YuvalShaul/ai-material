"""When showing beats explaining.

Some things are quicker to demonstrate than to describe. A format is the
classic case: three examples settle what a paragraph of rules leaves open.

    python 2-few-shot.py
"""

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"

INPUTS = ["Ms. Jane Q. Doe-Smith", "  bob jones ", "PROF. ALAN TURING"]

# Explaining the rule. Every clause here is an attempt to close a gap.
DESCRIBED = """Normalise each name: strip titles, collapse whitespace, keep
hyphens, use Title Case, keep middle initials with a full stop, and return
one name per line with no commentary."""

# Showing the rule. Shorter, and far harder to misread.
SHOWN = """Normalise each name. Follow these examples exactly.

Dr. MARY-ANNE O'Leary  ->  Mary-Anne O'Leary
   john  r. smith      ->  John R. Smith
SIR ISAAC NEWTON       ->  Isaac Newton

Return one name per line, nothing else."""


def ask(system):
    r = client.messages.create(
        model=MODEL, max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": "\n".join(INPUTS)}],
    )
    return "".join(b.text for b in r.content if b.type == "text").strip()


print("--- described ---")
print(ask(DESCRIBED))
print()
print("--- shown ---")
print(ask(SHOWN))

# Try adding a fourth input the examples do not cover — a name with a
# suffix, "James Bond Jr." — and see which prompt handles it. Examples
# are a specification, and like any specification they have edges.
