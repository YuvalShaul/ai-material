"""What one call actually costs, from the numbers the API hands back.

Two things to notice:
  1. Output tokens cost about five times what input tokens cost.
  2. You did not have to guess anything — usage comes back with the reply.

Prices are USD per million tokens, correct at the time of writing.
Check https://www.anthropic.com/pricing before trusting them.

    python 5-cost.py
"""

import anthropic

client = anthropic.Anthropic()

MODEL = "claude-opus-5"

PRICES = {                    # (input, output) USD per 1M tokens
    "claude-opus-5":    (5.00, 25.00),
    "claude-sonnet-5":  (2.00, 10.00),
    "claude-haiku-4-5": (1.00,  5.00),
}

response = client.messages.create(
    model=MODEL,
    max_tokens=2000,
    messages=[{"role": "user", "content": "Explain what an API is, in about 200 words."}],
)

in_price, out_price = PRICES[MODEL]
cost_in = response.usage.input_tokens / 1_000_000 * in_price
cost_out = response.usage.output_tokens / 1_000_000 * out_price

print(f"model  {MODEL}")
print(f"in     {response.usage.input_tokens:6d} tokens   ${cost_in:.6f}")
print(f"out    {response.usage.output_tokens:6d} tokens   ${cost_out:.6f}   <- ~5x per token")
print(f"total                   ${cost_in + cost_out:.6f}")
print()
print("Now imagine this call is turn 20 of a conversation, and every earlier")
print("turn is re-sent as input. That is the bill an agent runs up.")
