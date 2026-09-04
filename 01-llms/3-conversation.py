"""Memory is not a feature of the model. It is something you send.

Same two questions as 2-stateless.py. The only difference is that the
second call carries the first exchange along with it.

Watch the token count on the second call. You are paying to re-send
everything that was said before. That number only ever goes up.

    python 3-conversation.py
"""

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"


def ask(messages):
    response = client.messages.create(model=MODEL, max_tokens=1000, messages=messages)
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, response.usage


messages = [{"role": "user", "content": "My name is Yuval. Please remember it."}]

answer, usage = ask(messages)
print("--- call 1 ---")
print(answer)
print("sent:", usage.input_tokens, "tokens")

# Append the reply, then the next question. THIS is the memory.
messages.append({"role": "assistant", "content": answer})
messages.append({"role": "user", "content": "What is my name?"})

answer, usage = ask(messages)
print()
print("--- call 2 ---")
print(answer)
print("sent:", usage.input_tokens, "tokens  <- the whole conversation, again")
