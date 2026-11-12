import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app import ingest, search
from app.agent import build_graph

STATIC = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    search.ensure_indexes()
    yield


app = FastAPI(title="repochat", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class IndexRequest(BaseModel):
    url: str


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    repo: str
    messages: list[ChatMessage] = Field(min_length=1)


@app.get("/")
def index_page():
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/repos")
async def index_repo(body: IndexRequest):
    try:
        repo = ingest.parse_repo(body.url)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    job = await ingest.start_index(repo)
    return job


@app.get("/repos/{owner}/{name}")
def repo_status(owner: str, name: str):
    repo = f"{owner}/{name}".lower()
    job = ingest.JOBS.get(repo)
    if job:
        return job
    if search.repo_indexed(repo):
        return ingest.Job(repo=repo, status="ready")
    raise HTTPException(404, "Repo not indexed")


@app.post("/chat")
async def chat(body: ChatRequest):
    repo = body.repo.lower()
    job = ingest.JOBS.get(repo)
    if (job and job.status != "ready") or (not job and not search.repo_indexed(repo)):
        raise HTTPException(409, "Repo is not indexed yet")
    graph = build_graph(repo)
    history = [
        HumanMessage(m.content) if m.role == "user" else AIMessage(m.content) for m in body.messages
    ]

    def translate(event):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            text = event["data"]["chunk"].content
            if text:
                return {"event": "token", "data": json.dumps(text)}
        elif kind == "on_tool_start":
            return {
                "event": "tool",
                "data": json.dumps({"name": event["name"], "input": event["data"].get("input")}),
            }
        return None

    async def events():
        try:
            async for event in graph.astream_events(
                {"messages": history, "repo": repo},
                version="v2",
                config={"recursion_limit": 16},
            ):
                if out := translate(event):
                    yield out
        except GraphRecursionError:
            yield {
                "event": "token",
                "data": json.dumps(
                    "\n\nI searched the repository several times and could not find a grounded answer."
                ),
            }
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(events())
