"""Asking for JSON, and then guaranteeing it.

"Reply only with JSON" works most of the time, which is the problem: the
failures arrive in production, wrapped in a code fence or preceded by
"Here you go!".

output_config makes the format part of the request rather than a request.

    python 3-structured-output.py
"""

import json
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"

TICKET = ("Subject: cannot log in since the weekend. I'm on the Team plan, "
          "account 44192. Tried three browsers. This is blocking my whole team.")

SCHEMA = {
    "type": "object",
    "properties": {
        "account_id": {"type": "string"},
        "plan": {"type": "string", "enum": ["Free", "Team", "Enterprise", "unknown"]},
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        "one_line_summary": {"type": "string"},
    },
    "required": ["account_id", "plan", "severity", "one_line_summary"],
    "additionalProperties": False,
}

# --- the hopeful way -------------------------------------------------
loose = client.messages.create(
    model=MODEL, max_tokens=1000,
    messages=[{"role": "user",
               "content": f"Extract account_id, plan, severity and a one line "
                          f"summary as JSON.\n\n{TICKET}"}],
)
loose_text = "".join(b.text for b in loose.content if b.type == "text")
print("--- asked politely ---")
print(loose_text.strip()[:300])
try:
    json.loads(loose_text)
    print("\n(parsed — this time)")
except json.JSONDecodeError as e:
    print(f"\njson.loads failed: {e}")

# --- the guaranteed way ----------------------------------------------
strict = client.messages.create(
    model=MODEL, max_tokens=1000,
    messages=[{"role": "user", "content": f"Extract the fields.\n\n{TICKET}"}],
    output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
)
text = next(b.text for b in strict.content if b.type == "text")
data = json.loads(text)          # no try/except needed: the format is enforced
print("\n--- with output_config ---")
print(json.dumps(data, indent=2))
print("\nseverity is one of the enum values:", data["severity"])

# There is a shorter form if you use Pydantic: client.messages.parse(...,
# output_format=YourModel) validates into a real object for you.
