"""The same shape, with the weights on your own machine.

Needs Ollama running, and one model pulled:

    ollama serve
    ollama pull llama3.2

Standard library only — no SDK, no framework. The point is that this is
the same function as 1-call.py: a string goes in, a string comes back.
What changed is where the weights live, not what the thing does.

    python 6-local-call.py
"""

import json
import urllib.request

payload = {
    "model": "llama3.2",
    "prompt": "In one sentence: what is a token?",
    "stream": False,
}

request = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(request) as response:
    body = json.load(response)

print(body["response"].strip())
print()
print("prompt tokens:", body.get("prompt_eval_count"))
print("output tokens:", body.get("eval_count"))

# No API key. No bill. No network call leaving your machine.
# In exchange: a smaller model, and your laptop does the work.
