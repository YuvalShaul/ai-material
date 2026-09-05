# LangChain 1.x reference notes

Three files, one per package, covering the pieces you actually touch when building an agent with
the LangChain 1.x stack. Every entry says what the class or function *defines or does*, links to the
official API page (opens in a new tab), and ends with a runnable example whose printed output was
verified against the pinned versions.

| File | Package | Pinned version | What the package contributes |
| --- | --- | --- | --- |
| [`langchain_core-reference.md`](langchain_core-reference.md) | `langchain-core` | 1.6.2 | The **types**: message classes, `BaseChatModel`, `BaseTool`/`@tool`, `Runnable`, prompts, output parsers. No provider client, no loop, no runtime. |
| [`langgraph-reference.md`](langgraph-reference.md) | `langgraph` | 1.2.11 | The **runtime**: `StateGraph` → `CompiledStateGraph`, state with reducers, `Command`, `interrupt`, checkpointers, `Send`. Knows nothing about models. |
| [`langchain-reference.md`](langchain-reference.md) | `langchain` | 1.4.0 | The **prebuilt agent**: `create_agent`, `AgentState`, middleware, structured output, `init_chat_model`. Assembles a langgraph graph out of langchain-core types. |

## How the three parts relate

```
              you write                          you run
                 │                                  │
   langchain     │  create_agent(model, tools, middleware=…, response_format=…)
   (agent layer) │        │ builds a StateGraph and calls .compile()
                 ▼        ▼
   langgraph     │  CompiledStateGraph.invoke / stream / get_state
   (runtime)     │        │ runs nodes, merges state, checkpoints, pauses on interrupt()
                 ▼        ▼
   langchain-core│  BaseChatModel.invoke(list[BaseMessage]) -> AIMessage
   (types)       │  BaseTool.invoke(tool_call) -> ToolMessage
                 │  HumanMessage / AIMessage / ToolMessage / SystemMessage
                 ▼
   provider pkg  │  langchain-anthropic, langchain-openai, … subclass BaseChatModel
```

**The dependency arrow points down and only down.** Verified from the installed packages'
metadata and by import tracing:

- `langchain` 1.4.0 requires `langgraph<1.3.0,>=1.2.11` and `langchain-core<2.0.0,>=1.6.0`.
- `langgraph` 1.2.11 requires `langchain-core<2,>=1.4.7`. It does **not** require `langchain`, and
  importing it does not import `langchain`.
- `langchain-core` 1.6.2 requires neither. Importing every one of its modules leaves `langgraph`
  and `langchain` absent from `sys.modules`.

What that means in practice:

- **An agent is a graph.** `create_agent` (langchain) returns a `CompiledStateGraph` (langgraph).
  Everything the langgraph file says about `invoke`, `stream`, `get_state`, checkpointers and
  `interrupt` applies to an agent unchanged, and `agent.get_graph().draw_mermaid()` shows you the
  `model ⇄ tools` loop it built.
- **The graph's state is a list of langchain-core messages.** `AgentState.messages` (langchain) is
  `MessagesState.messages` (langgraph) is `list[AnyMessage]` merged by `add_messages`, and the
  items are `HumanMessage`/`AIMessage`/`ToolMessage` (langchain-core).
- **Models and tools are langchain-core objects.** `create_agent` calls `BaseChatModel.bind_tools`
  and runs `BaseTool.invoke(tool_call)`; the provider package supplies the concrete model class.
- **Middleware is the seam between the layers.** `AgentMiddleware` hooks (langchain) receive the
  langgraph state and a `ModelRequest` holding langchain-core messages and tools.

## Which file to open

| You are holding / deciding… | Open |
| --- | --- |
| a message, a tool, a model object, a prompt template, `\|` pipelines | `langchain_core-reference.md` |
| what runs next, how state is merged, pausing/resuming, parallel fan-out, persistence | `langgraph-reference.md` |
| "give me an agent without drawing the graph", approval before tools, context trimming, structured answers, `"provider:model"` strings | `langchain-reference.md` |

Each file has the same shape: pinned version → what the package is → an **Index** table
(name, what it defines/does, most-used members) → one section per name with the official link, a
concrete description, a verified example, and for graphs the Mermaid diagram the code itself
printed.

## Reproducing the examples

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "langchain==1.4.0" "langchain-core==1.6.2" "langgraph==1.2.11" "langchain-anthropic==1.7.1"
```

All examples run offline (they use `langchain_core`'s `GenericFakeChatModel` or a five-line
`BaseChatModel` subclass) except one, marked **The one live example** in the langchain file, which
needs `ANTHROPIC_API_KEY` and skips itself without it. Verified on Python 3.13.7 on 2026-09-05.

## Old-tutorial warnings, collected

- `langgraph.prebuilt.create_react_agent` → deprecated since LangGraph 1.0 (warning class
  `LangGraphDeprecatedSinceV10`, removal planned for 2.0). Use `langchain.agents.create_agent`.
- `AgentExecutor`, `initialize_agent`, `LLMChain`, `ConversationChain` and the other `Chain`
  classes → moved to the separate `langchain-classic` package (1.0.8); importing them from
  `langchain.agents` raises `ImportError`. A tutorial using them predates 1.0.
- `langchain-mcp-adapters` → its tool conversion was ported into `langchain.mcp` (`MCPAdapter`) in
  langchain 1.4.0; needs the `langchain[mcp]` extra.

## What changed since the previous edition of these notes

The previous edition pinned `langchain==1.3.14`, `langchain-core==1.5.3`, `langgraph==1.2.10`.
All 28 of its examples produce identical output on the versions pinned here. Notable additions in
between, none of which change the APIs documented in the three files:

- **langchain 1.3.14 → 1.4.0:** the `langchain.mcp` namespace (`MCPAdapter`, `as_langchain_tool`);
  `ContextEditingMiddleware` accepts a custom `token_counter`; standard model exception types.
- **langchain-core 1.5.3 → 1.6.2:** standard model exception types; `StructuredTool` is
  JSON-serializable; assorted fixes to tool-schema generation and content-block handling.
- **langgraph 1.2.10 → 1.2.11:** `trace_policy` on `add_node`; `langgraph-checkpoint` 4.2.0.
