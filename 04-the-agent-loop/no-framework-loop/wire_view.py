"""Printing what goes over the wire: one table per call, one row per block.

A row marked -> was sent, a row marked <- came back. The three files in
this directory share this so each of them stays about its own subject:
the protocol, the loop, the tools.

Nothing here talks to the API. It only prints.
"""

import json


def print_messages(messages, tools=None):
    """The request: the list, one row per block, with the tool schemas.

    The tools are a top-level argument rather than a message, so they have
    no role of their own; they ride along with the first row. They go up on
    every call.
    """
    print(f"   {'role':<9} | blocks")
    print(f"{'-' * 12}-+-{'-' * 58}")
    first = True
    for message in messages:
        label = f"{arrow(message['role'])} {message['role']}"
        if first and tools:
            print(f"{label:<12} | tools: {tool_names(tools)}")
            label = ""
        print_msg_blocks(label, message["content"])
        first = False


def arrow(role):
    """Which side the message came from: -> your code, <- the model.

    An assistant message keeps its <- when the list is re-sent, because the
    model is still the one that wrote it.
    """
    return "<-" if role == "assistant" else "->"


def tool_names(tools):
    """Just the names: what the model is choosing from."""
    names = []
    for tool in tools:
        names.append(tool["name"])
    return ", ".join(names)


def print_reply(reply, tokens=False):
    """What came back: the boundary line, then the blocks and the stop_reason."""
    print(f"{'- ' * 6}-+{'- ' * 29}".rstrip())
    print_msg_blocks(f"{arrow(reply.role)} {reply.role}", reply.content)
    print(f"{'':<12} | stop_reason={reply.stop_reason}")
    if tokens:
        print(f"{'':<12} | sent {reply.usage.input_tokens} tokens, "
              f"got {reply.usage.output_tokens}")
    print()


def print_msg_blocks(label, content):
    """One message: its label on the first row, then one row per block."""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    first = True
    for block in content:
        print_block(label if first else "", block)
        first = False


def print_block(label, block):
    """One row: the label column, then the block's main fields."""
    if not isinstance(block, dict):
        block = block.model_dump()
    kind = block["type"]
    if kind == "text":
        fields = f'text "{shorten(block["text"])}"'
    elif kind == "tool_use":
        # the arguments are shown: one reply can hold several calls to the
        # same tool, and only the arguments tell them apart
        fields = f'tool_use {block["name"]}({json.dumps(block["input"])}) #{short_id(block["id"])}'
    elif kind == "tool_result":
        fields = (f'tool_result "{shorten(block["content"], count_lines=True)}" '
                  f'#{short_id(block["tool_use_id"])}')
    else:
        fields = kind
    print(f"{label:<12} | {fields}")


def shorten(text, limit=52, count_lines=False):
    """One line short enough for a row.

    Tool output can be a whole file, so count_lines reports its size
    instead of its first words. Prose is always shown, cut to fit.
    """
    text = text.strip()
    if count_lines and len(text) > limit and "\n" in text:
        return f"{len(text.splitlines())} lines"
    text = text.replace("\n", " ")
    if len(text) > limit:
        text = text[:limit - 3] + "..."
    return text


def short_id(block_id):
    """A tool id, cut to the part that is worth reading. The pairing still shows."""
    return block_id[6:14]
