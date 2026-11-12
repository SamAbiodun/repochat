from typing import Annotated

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from app import search
from app.llm import chat_model, embeddings

MAX_TOOL_CALLS = 6

SYSTEM = """You are repochat, an assistant that answers questions about one GitHub repository: {repo}.
Ground every answer in the repository. Use the tools before answering anything about the code.
Start with search_code. If the results look thin or off-topic, search again with different words,
or use grep_code for exact identifiers, list_files to orient, and read_file for full context.
Cite sources as path:line (for example `app/main.py:42`) next to each claim.
If the repository does not contain the answer, say so plainly instead of guessing.
Use at most six tool calls, then answer with what you have.
Answer in at most three short sentences, or a short list of at most five items.
Lead with the answer. Quote at most one small code snippet, only when it is the answer itself.
No preamble, no summary of what you searched, no closing remarks."""


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    repo: str


def build_tools(repo: str):
    @tool
    def search_code(query: str) -> str:
        """Semantic plus keyword search over code and docs. Use natural language or identifiers."""
        vector = embeddings().embed_query(query)
        hits = search.hybrid_search(repo, query, vector)
        if not hits:
            return "No results."
        return "\n\n".join(
            f"{h['path']}:{h['start_line']}-{h['end_line']}\n{h['content']}" for h in hits
        )

    @tool
    def grep_code(pattern: str) -> str:
        """Exact or regex match over every file, like grep -rin. Best for identifiers and strings."""
        hits = search.grep(repo, pattern)
        if not hits:
            return "No matches."
        return "\n".join(f"{h['path']}:{h['line']}: {h['text']}" for h in hits)

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
        """Read a file by its repository path, optionally a line range."""
        text = search.get_file(repo, path)
        if text is None:
            return f"No such file: {path}"
        lines = text.splitlines()
        start = max(start_line, 1)
        end = min(end_line, len(lines))
        return "\n".join(
            f"{n}: {line}" for n, line in enumerate(lines[start - 1 : end], start=start)
        )

    @tool
    def list_files(prefix: str = "") -> str:
        """List file paths in the repository, optionally under a directory prefix."""
        paths = [p for p in search.list_paths(repo) if p.startswith(prefix)]
        return "\n".join(paths[:400]) or "No files."

    return [search_code, grep_code, read_file, list_files]


def build_graph(repo: str):
    tools = build_tools(repo)
    base = chat_model()
    with_tools = base.bind_tools(tools)

    def call_model(state: State):
        messages = [SystemMessage(SYSTEM.format(repo=repo)), *state["messages"]]
        used = sum(isinstance(m, ToolMessage) for m in state["messages"])
        if used >= MAX_TOOL_CALLS:
            messages.append(
                HumanMessage(
                    "Tool budget is spent. Answer now from what you found. "
                    "If it was not there, say so in one sentence."
                )
            )
            return {"messages": [base.invoke(messages)]}
        return {"messages": [with_tools.invoke(messages)]}

    g = StateGraph(State)
    g.add_node("agent", call_model)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()
