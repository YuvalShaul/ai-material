# `langchain-core` Reference

**Pinned version: `langchain-core==1.6.2`** (released 2026-09-04). Everything below was derived by
inspecting this exact installed package, and every example was executed against it on Python 3.13.
Each heading links to <a href="https://reference.langchain.com/python/langchain-core" target="_blank" rel="noopener">reference.langchain.com</a>,
which is unversioned and always serves the latest 1.x. Links open in a new tab.

Part of a three-file set — see the [README](README.md) for how `langchain-core`,
`langgraph` and `langchain` fit together.

## What `langchain-core` is

`langchain-core` is the package that defines the *data types* the rest of the ecosystem passes
around, and the *base classes* the other packages subclass. Concretely, it ships:

| Module | What it defines |
| --- | --- |
| `langchain_core.messages` | `HumanMessage`, `AIMessage`, `ToolMessage`, `SystemMessage`, the `tool_calls` dict shape, `content_blocks` |
| `langchain_core.language_models` | `BaseChatModel` (plus fakes such as `GenericFakeChatModel` for tests) |
| `langchain_core.tools` | `BaseTool`, `StructuredTool`, the `@tool` decorator, `ToolException` |
| `langchain_core.runnables` | `Runnable`, `RunnableLambda`, `RunnableSequence` (what `\|` builds), `RunnableConfig` |
| `langchain_core.prompts` | `ChatPromptTemplate`, `MessagesPlaceholder`, `PromptTemplate` |
| `langchain_core.output_parsers` | `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser` |
| `langchain_core.callbacks`, `.tracers` | The callback/tracing hooks LangSmith plugs into |

What it does **not** ship: an HTTP client for any model provider (that is `langchain-anthropic`,
`langchain-openai`, …), an agent loop (that is `langchain`), a graph runtime or checkpointer (that
is `langgraph`). A LangGraph agent can accept a chat model from any provider because every provider's
class subclasses the `BaseChatModel` defined here, and every node exchanges the message classes
defined here.

**Layer order — the dependency arrow points one way:**

```
langchain  ──depends on──▶  langgraph  ──depends on──▶  langchain-core
```

Verified from package metadata: `langchain` 1.4.0 requires `langgraph<1.3.0,>=1.2.11` and
`langchain-core<2.0.0,>=1.6.0`; `langgraph` 1.2.11 requires `langchain-core<2,>=1.4.7` and **does not
require `langchain` at all**; `langchain-core` 1.6.2 requires neither (its own dependencies are
`pydantic`, `langsmith`, `langchain-protocol`, `httpx`, `tenacity`, `jsonpatch`, `pyyaml`,
`typing-extensions`, `packaging`, `uuid-utils`). Importing every module of `langchain_core` leaves
`langgraph` and `langchain` absent from `sys.modules`.

## Index

| Name | What it defines / does | Most-used methods & fields |
| --- | --- | --- |
| [`BaseChatModel`](#basechatmodel) | Abstract class: a subclass implements `_generate` and `_llm_type`; in return it inherits `invoke`/`stream`/`batch`, tool binding and structured output | `invoke`, `stream`, `bind_tools`, `with_structured_output` |
| [`HumanMessage`](#humanmessage) | Message with `type="human"`; `content` is a string or a list of content blocks; `.text` flattens either | `content`, `text`, `content_blocks` |
| [`AIMessage`](#aimessage) | Message with `type="ai"` plus `tool_calls` (what the model wants run), `invalid_tool_calls` and `usage_metadata` | `text`, `tool_calls`, `usage_metadata` |
| [`ToolMessage`](#toolmessage) | Message with `type="tool"`; `tool_call_id` pairs it with the `AIMessage.tool_calls` entry it answers; `status` is `"success"` or `"error"` | `content`, `tool_call_id`, `status`, `artifact` |
| [`SystemMessage`](#systemmessage) | Message with `type="system"`; providers send it as the system prompt, outside the user/assistant turns | `content`, `text` |
| [`@tool`](#tool) | Builds a `StructuredTool` from a function: name from the function name, description from the docstring, argument schema from the type hints | `invoke`, `name`, `description`, `args` |
| [`BaseTool`](#basetool) | Abstract `Runnable`: a subclass implements `_run`; it gets `invoke` (dict or tool-call in, `ToolMessage` out), schema generation and error handling | `invoke`, `args`, `tool_call_schema`, `handle_tool_error` |
| [`Runnable`](#runnable) | The calling convention shared by models, tools, prompts and parsers: `invoke`/`batch`/`stream`, async twins, `\|` composition, retries and fallbacks | `invoke`, `batch`, `stream`, `pipe` (`\|`), `with_retry` |
| [`ChatPromptTemplate`](#chatprompttemplate) | Holds `(role, template)` pairs; `format_messages(**vars)` fills `{placeholders}` and returns message objects; `MessagesPlaceholder` splices a history list in | `from_messages`, `format_messages`, `partial`, `invoke` |
| [`StrOutputParser`](#stroutputparser) | `Runnable` that takes a message (or `str`) and returns its `.text`, so a pipeline ends in a plain string | `invoke`, `parse` |

---

### <a href="https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel" target="_blank" rel="noopener"><code>BaseChatModel</code></a>

The abstract base class every chat model subclasses. It leaves exactly two members abstract:
`_generate(messages, stop, run_manager, **kwargs) -> ChatResult` (one model call) and the `_llm_type`
property (a string naming the provider). Everything else is inherited:

- **Input normalisation.** `invoke` accepts a `str` (wrapped in a `HumanMessage`), a list of
  messages, a list of `(role, text)` tuples or dicts, or a `PromptValue`, and hands `_generate` a
  list of `BaseMessage` objects. It returns one `AIMessage`.
- **`Runnable` methods.** `invoke`, `batch`, `stream` and the async `ainvoke`/`abatch`/`astream`.
  If the subclass does not override `_stream`, `stream` falls back to calling `_generate` once and
  yielding a single chunk.
- **`bind_tools(tools, tool_choice=None)`** — returns a new `Runnable` with tool schemas attached
  to each request, so the model can answer with `tool_calls`. The base implementation raises
  `NotImplementedError`; every provider package overrides it.
- **`with_structured_output(schema, include_raw=False)`** — returns a `Runnable` whose output is a
  dict or Pydantic object matching `schema` instead of an `AIMessage`. Also provider-implemented.
- **Bookkeeping fields:** `cache`, `rate_limiter`, `callbacks`, `tags`, `metadata`,
  `disable_streaming`, and `get_num_tokens_from_messages` (a rough default; providers override it).

You never instantiate this class directly — a provider package gives you a concrete subclass
(`ChatAnthropic`, `ChatOpenAI`, …). The example below implements the two abstract members to show
that this is all a chat model must provide.

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class EchoModel(BaseChatModel):
    """The two members BaseChatModel leaves abstract — nothing else is required."""

    @property
    def _llm_type(self) -> str:
        return "echo"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        reply = AIMessage(f"You said: {messages[-1].text}")
        return ChatResult(generations=[ChatGeneration(message=reply)])


model = EchoModel()

reply = model.invoke([SystemMessage("Be terse."), HumanMessage("Capital of France?")])
print(type(reply).__name__, "|", reply.text)      # AIMessage | You said: Capital of France?

print(model.invoke("hi").text)                     # You said: hi   <- a str becomes a HumanMessage
print([r.text for r in model.batch(["a", "b"])])   # ['You said: a', 'You said: b']
print([c.text for c in model.stream("hi")])        # ['You said: hi']  <- no _stream, so one chunk

try:
    model.bind_tools([])
except NotImplementedError:
    print("bind_tools: not implemented here — providers override it")
```

`langchain_core.language_models.GenericFakeChatModel` is core's own scripted test double (you hand
it an iterator of replies); the other two references use it so their examples run offline.

[↩ back to index](#index)

---

### <a href="https://reference.langchain.com/python/langchain-core/messages/human/HumanMessage" target="_blank" rel="noopener"><code>HumanMessage</code></a>

The user's turn. Fields it shares with every message (`BaseMessage`): `content` — a `str`, or a
list of content-block dicts for multimodal input (`{"type": "text", ...}`, `{"type": "image", ...}`);
`id` — optional, used by LangGraph's `add_messages` to replace a message instead of appending;
`name`; `additional_kwargs` and `response_metadata` (provider-specific extras). Its `type` is fixed
to `"human"`, which is what a provider integration reads to map it onto its `user` role.

Two properties normalise the shape of `content`: `.text` returns the concatenated text however
`content` is stored, and `.content_blocks` returns it as a list of typed blocks even when it was a
plain string.

```python
from langchain_core.messages import HumanMessage

msg = HumanMessage("What is 2 + 2?")
print(msg.type, "|", msg.text)     # human | What is 2 + 2?
print(msg.content_blocks)          # [{'type': 'text', 'text': 'What is 2 + 2?'}]

multimodal = HumanMessage(content=[{"type": "text", "text": "Describe this"},
                                   {"type": "image", "url": "https://example.com/cat.png"}])
print(multimodal.text)             # Describe this   <- .text skips the non-text block
```

[↩ back to index](#index)

---

### <a href="https://reference.langchain.com/python/langchain-core/messages/ai/AIMessage" target="_blank" rel="noopener"><code>AIMessage</code></a>

The model's turn, and the return type of `BaseChatModel.invoke`. On top of the shared fields it
adds the three that make agent loops possible:

- `tool_calls` — a list of dicts `{"name", "args", "id", "type": "tool_call"}`, one per tool the
  model wants executed. `args` is already parsed into a Python dict. When a model calls tools,
  `content` is usually empty and this list is the real payload.
- `invalid_tool_calls` — calls whose arguments the provider returned but core could not parse as
  JSON; each carries the raw string and an `error`.
- `usage_metadata` — `{"input_tokens", "output_tokens", "total_tokens"}` plus optional
  `input_token_details`/`output_token_details` (cache reads, reasoning tokens).

An agent loop reads `tool_calls`, runs each one, and answers every `id` with a `ToolMessage`.

```python
from langchain_core.messages import AIMessage

msg = AIMessage(
    content="",
    tool_calls=[{"name": "add", "args": {"a": 2, "b": 2}, "id": "call_1"}],
    usage_metadata={"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
)
print(msg.type, "|", msg.tool_calls[0]["name"], msg.tool_calls[0]["args"])
# ai | add {'a': 2, 'b': 2}
print(msg.tool_calls[0]["type"], "|", msg.usage_metadata["total_tokens"])
# tool_call | 19
```

[↩ back to index](#index)

---

### <a href="https://reference.langchain.com/python/langchain-core/messages/tool/ToolMessage" target="_blank" rel="noopener"><code>ToolMessage</code></a>

The result of running one tool call, appended to the conversation so the model can continue. What
it adds to the shared fields:

- `tool_call_id` (required) — must equal the `id` of the `AIMessage.tool_calls` entry it answers;
  providers reject a conversation where a call has no matching result.
- `status` — `"success"` (default) or `"error"`. `BaseTool.invoke` sets `"error"` when the tool
  raised and `handle_tool_error` is on; the model sees the error text as `content`.
- `artifact` — anything you want to keep for the program but *not* send to the model (a DataFrame,
  raw bytes, a file handle). Populated when a tool uses `response_format="content_and_artifact"`.

```python
from langchain_core.messages import ToolMessage

ok = ToolMessage(content="4", tool_call_id="call_1")
print(ok.type, "|", ok.tool_call_id, "|", ok.status)           # tool | call_1 | success

failed = ToolMessage(content="division by zero", tool_call_id="call_2", status="error")
print(failed.status, "|", failed.text)                          # error | division by zero

with_artifact = ToolMessage(content="3 rows", tool_call_id="call_3", artifact=[1, 2, 3])
print(with_artifact.text, "|", with_artifact.artifact)          # 3 rows | [1, 2, 3]
```

[↩ back to index](#index)

---

### <a href="https://reference.langchain.com/python/langchain-core/messages/system/SystemMessage" target="_blank" rel="noopener"><code>SystemMessage</code></a>

Standing instructions — persona, rules, output format. Its `type` is `"system"`. Provider
integrations pull it out of the message list and send it through the provider's system-prompt
channel (Anthropic's top-level `system` parameter, OpenAI's `system`/`developer` role) rather than
as a conversational turn, which is why it conventionally sits first and appears once.
`langchain.agents.create_agent(system_prompt=...)` accepts either a string or a `SystemMessage`.

```python
from langchain_core.messages import HumanMessage, SystemMessage

conversation = [SystemMessage("You are a terse calculator."), HumanMessage("2 + 2?")]
print([m.type for m in conversation])   # ['system', 'human']
```

[↩ back to index](#index)

---

### <a href="https://reference.langchain.com/python/langchain-core/tools/convert/tool" target="_blank" rel="noopener"><code>@tool</code></a>

A decorator that builds a `StructuredTool` (a `BaseTool` subclass) from a plain function. What it
reads from the function, and what the model then sees when deciding whether to call it:

- **name** — the function name, unless you pass one: `@tool("add_numbers")`.
- **description** — the docstring (`description=` overrides). It is sent to the model with every
  request, so it is part of the prompt.
- **argument schema** — generated from the type hints and defaults. `parse_docstring=True` also
  reads a Google-style `Args:` section into per-argument descriptions. `args_schema=` replaces the
  inference with a Pydantic model of your own.

Other switches: `return_direct=True` tells an agent loop to return the tool's output to the user
without another model call; `response_format="content_and_artifact"` lets the function return a
`(content, artifact)` pair, where only `content` reaches the model.

```python
from langchain_core.tools import tool


@tool(parse_docstring=True)
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First addend.
        b: Second addend.
    """
    return a + b


print(type(add).__name__, "|", add.name, "|", add.description)
# StructuredTool | add | Add two integers.
print(add.args)
# {'a': {'description': 'First addend.', 'title': 'A', 'type': 'integer'},
#  'b': {'description': 'Second addend.', 'title': 'B', 'type': 'integer'}}
print(add.invoke({"a": 2, "b": 3}))     # 5
```

[↩ back to index](#index)

---

### <a href="https://reference.langchain.com/python/langchain-core/tools/base/BaseTool" target="_blank" rel="noopener"><code>BaseTool</code></a>

The abstract class behind every tool, including the ones `@tool` builds. A subclass declares
`name` and `description` as fields and implements one method, `_run(**args)`; `_arun` is optional
(the default runs `_run` in a thread). In return it gets:

- **Two ways to be invoked.** `invoke({"a": 1})` runs the tool and returns its raw result.
  `invoke(tool_call_dict)` — the dict straight from `AIMessage.tool_calls` — runs it and returns a
  ready-made `ToolMessage` carrying the matching `tool_call_id`. Agent loops use the second form.
- **Schema.** `args` (the JSON-schema properties) and `tool_call_schema` (a Pydantic model), derived
  from `_run`'s signature or from `args_schema`; `bind_tools` sends this to the model.
- **Error handling.** With `handle_tool_error=True`, a `ToolException` raised inside `_run` becomes
  a `ToolMessage` with `status="error"` instead of propagating; a string or callable there
  customises the error text. `handle_validation_error` does the same for bad arguments.

Subclass it directly when a tool needs state or setup (a connection, a cache) that a bare function
cannot hold; otherwise `@tool` is shorter.

```python
from langchain_core.tools import BaseTool, ToolException


class Echo(BaseTool):
    name: str = "echo"
    description: str = "Echo the input text."

    def _run(self, text: str) -> str:
        return text


class Boom(BaseTool):
    name: str = "boom"
    description: str = "Always fails."
    handle_tool_error: bool = True

    def _run(self, text: str) -> str:
        raise ToolException("no can do")


print(Echo().invoke({"text": "hello"}))                                 # hello
print(Echo().tool_call_schema.model_json_schema()["properties"])        # {'text': {'title': 'Text', 'type': 'string'}}

call = {"name": "echo", "args": {"text": "hi"}, "id": "c1", "type": "tool_call"}
print(repr(Echo().invoke(call)))
# ToolMessage(content='hi', name='echo', tool_call_id='c1')

call = {"name": "boom", "args": {"text": "x"}, "id": "c2", "type": "tool_call"}
result = Boom().invoke(call)
print(result.status, "|", result.content)                               # error | no can do
```

[↩ back to index](#index)

---

### <a href="https://reference.langchain.com/python/langchain-core/runnables/base/Runnable" target="_blank" rel="noopener"><code>Runnable</code></a>

The one calling convention in core. Chat models, tools, prompts, parsers — and LangGraph's compiled
graphs — all subclass it, which is why they are called the same way and can be chained. What it
defines:

- **Execution:** `invoke(input, config=None)`, `batch(inputs)`, `stream(input)` (a generator), and
  the async twins `ainvoke`, `abatch`, `astream`. A subclass implements `invoke` and the rest
  derive from it unless overridden (`batch` runs in a thread pool; `stream` yields one item).
- **Composition:** `a | b` builds a `RunnableSequence` that feeds `a`'s output to `b`; `pipe()` is
  the method form. A plain function or a dict of runnables on either side of `|` is coerced
  (`RunnableLambda`, `RunnableParallel`).
- **Behaviour wrappers**, each returning a new `Runnable`: `with_retry(stop_after_attempt=…)`,
  `with_fallbacks([other])`, `with_config(tags=…, callbacks=…)`, `bind(**kwargs)`.
- **Introspection:** `get_input_schema()`, `get_output_schema()`, `get_graph()` (the same object
  `CompiledStateGraph.get_graph()` returns), `astream_events()` for a token-by-token event stream.

```python
import asyncio

from langchain_core.runnables import RunnableLambda

shout = RunnableLambda(lambda s: s.upper())
exclaim = RunnableLambda(lambda s: s + "!")
chain = shout | exclaim

print(type(chain).__name__)                          # RunnableSequence
print(chain.invoke("hello"))                         # HELLO!
print(chain.batch(["a", "b"]))                       # ['A!', 'B!']
print(list(chain.stream("hey")))                     # ['HEY!']
print(asyncio.run(chain.ainvoke("async")))           # ASYNC!

safe = (shout | RunnableLambda(lambda s: 1 / 0)).with_fallbacks([exclaim])
print(safe.invoke("fallback"))                       # fallback!   <- the failing chain fell back
```

[↩ back to index](#index)

---

### <a href="https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate" target="_blank" rel="noopener"><code>ChatPromptTemplate</code></a>

A list of message templates that becomes a list of messages once variables are supplied.
`from_messages` takes `(role, template)` tuples — roles `"system"`, `"human"`, `"ai"` — where
`{name}` placeholders are the variables; `input_variables` lists what it found. Filling it:

- `format_messages(**vars)` returns message objects; `invoke({"k": v})` does the same through the
  `Runnable` interface, which is how a template becomes the first stage of `prompt | model`.
- `MessagesPlaceholder("history")` is a slot that splices an entire list of messages in — the
  standard way to insert prior conversation turns.
- `partial(**vars)` pre-fills some variables and returns a template needing the rest.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You translate to {language}."),
    MessagesPlaceholder("history"),
    ("human", "{text}"),
])
print(prompt.input_variables)                     # ['history', 'language', 'text']

french = prompt.partial(language="French")        # one variable fewer to supply
messages = french.format_messages(history=[("human", "bonjour?"), ("ai", "bonjour")], text="hello")
for m in messages:
    print(m.type, "|", m.text)
# system | You translate to French.
# human | bonjour?
# ai | bonjour
# human | hello
```

[↩ back to index](#index)

---

### <a href="https://reference.langchain.com/python/langchain-core/output_parsers/string/StrOutputParser" target="_blank" rel="noopener"><code>StrOutputParser</code></a>

A `Runnable` whose `invoke` accepts a `BaseMessage` or a `str` and returns a `str`: for a message it
returns `.text`, for a string the string itself. Its only job is to end a pipeline so the caller
gets plain text instead of an `AIMessage`; it also works on streams, passing each chunk's text
through. Core's other parsers follow the same `BaseOutputParser` shape but return structure:
`JsonOutputParser` parses the reply as JSON (tolerating partial JSON while streaming), and
`PydanticOutputParser(pydantic_object=…)` validates it into a model and adds
`get_format_instructions()` for the prompt.

```python
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([("human", "{text}")])
model = GenericFakeChatModel(messages=iter([AIMessage("bonjour")]))
pipeline = prompt | model | StrOutputParser()

print(repr(pipeline.invoke({"text": "hello"})))          # 'bonjour'
print(repr(StrOutputParser().invoke(AIMessage("x"))))   # 'x'
print(repr(StrOutputParser().invoke("already text")))   # 'already text'
```

[↩ back to index](#index)
