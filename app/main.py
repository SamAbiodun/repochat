import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app import auth, ingest, search, store
from app.agent import build_graph

STATIC = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    search.ensure_indexes()
    yield


app = FastAPI(title="repochat", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.include_router(auth.router)
User = Annotated[dict, Depends(auth.require_user)]


class IndexRequest(BaseModel):
    url: str


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    repo: str
    messages: list[ChatMessage] = Field(min_length=1)


class SessionBody(BaseModel):
    repo: str
    title: str
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
                    "I searched the repository several times and could not find a grounded answer."
                ),
            }
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(events())


@app.get("/sessions")
def sessions_list(user: User):
    return store.list_sessions(user["id"])


@app.get("/sessions/{session_id}")
def sessions_get(session_id: str, user: User):
    doc = store.get_session(user["id"], session_id)
    if not doc:
        raise HTTPException(404, "no such session")
    return doc


@app.put("/sessions/{session_id}")
async def sessions_put(session_id: str, request: Request, body: SessionBody, user: User):
    if len(await request.body()) > store.MAX_SESSION_BYTES:
        raise HTTPException(413, "session too large")
    if not session_id.isalnum() or len(session_id) > 32:
        raise HTTPException(400, "bad session id")
    return store.put_session(
        user["id"], session_id, body.repo, body.title, [m.model_dump() for m in body.messages]
    )


@app.delete("/sessions/{session_id}")
def sessions_delete(session_id: str, user: User):
    store.delete_session(user["id"], session_id)
    return {"ok": True}
