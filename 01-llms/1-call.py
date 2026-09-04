"""A model call with nothing around it.

No framework. No loop. No tools. No memory. No agent.
A string goes up, a string comes back — that is the whole thing.

Everything else in this course is built on top of these fifteen lines.

    python 1-call.py
"""

import anthropic

client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY

question = "In one sentence: what is a context window?"

response = client.messages.create(
    model="claude-opus-5",
    max_tokens=1000,
    messages=[{"role": "user", "content": question}],
)

# response.content is a LIST of blocks, not a string. Ask for the text ones.
for block in response.content:
    if block.type == "text":
        print(block.text)

print()
print("tokens in :", response.usage.input_tokens)
print("tokens out:", response.usage.output_tokens)
