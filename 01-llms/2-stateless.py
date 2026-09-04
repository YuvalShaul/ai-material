"""The model has no memory.

Two calls. The second one has never heard of the first.

    python 2-stateless.py
"""

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"


def ask(messages):
    response = client.messages.create(model=MODEL, max_tokens=1000, messages=messages)
    return "".join(b.text for b in response.content if b.type == "text")


print("--- call 1 ---")
print(ask([{"role": "user", "content": "My name is Yuval. Please remember it."}]))

print()
print("--- call 2: a brand new call ---")
print(ask([{"role": "user", "content": "What is my name?"}]))

# It cannot know. Nothing carried over between the two calls.
# There is no session on the other side. There is no "conversation" object.
