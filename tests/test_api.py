from fastapi.testclient import TestClient

from app import ingest, main, search


def test_index_and_status(monkeypatch):
    monkeypatch.setattr(search, "ensure_indexes", lambda: None)
    monkeypatch.setattr(search, "repo_indexed", lambda repo: False)
    started: list[str] = []

    async def fake_start(repo):
        started.append(repo)
        ingest.JOBS[repo] = ingest.Job(repo=repo, status="indexing", files=3)
        return ingest.JOBS[repo]

    monkeypatch.setattr(ingest, "start_index", fake_start)
    with TestClient(main.app) as c:
        assert c.post("/repos", json={"url": "nope"}).status_code == 400
        r = c.post("/repos", json={"url": "https://github.com/Owner/Repo"})
        assert r.status_code == 200 and r.json()["repo"] == "owner/repo"
        assert started == ["owner/repo"]
        assert c.get("/repos/owner/repo").json()["files"] == 3
        assert c.get("/repos/x/y").status_code == 404
        assert (
            c.post(
                "/chat", json={"repo": "x/y", "messages": [{"role": "user", "content": "hi"}]}
            ).status_code
            == 409
        )


def test_sessions_require_sign_in(monkeypatch):
    monkeypatch.setattr(search, "ensure_indexes", lambda: None)
    with TestClient(main.app) as c:
        assert c.get("/auth/me").json()["user"] is None
        assert c.get("/sessions").status_code == 401
        assert (
            c.put("/sessions/abc", json={"repo": "x/y", "title": "t", "messages": []}).status_code
            == 401
        )


def test_sessions_round_trip(monkeypatch):
    monkeypatch.setattr(search, "ensure_indexes", lambda: None)
    user = {"id": "1", "login": "sam", "avatar": ""}
    monkeypatch.setattr(main.auth, "current_user", lambda request: user)
    docs: dict[str, dict] = {}
    monkeypatch.setattr(
        main.store,
        "put_session",
        lambda u, i, r, t, m: docs.setdefault(i, {"id": i, "repo": r, "title": t, "updated": 1}),
    )
    monkeypatch.setattr(main.store, "list_sessions", lambda u: list(docs.values()))
    monkeypatch.setattr(main.store, "delete_session", lambda u, i: docs.pop(i, None))
    with TestClient(main.app) as c:
        body = {"repo": "x/y", "title": "t", "messages": [{"role": "user", "content": "hi"}]}
        assert c.put("/sessions/abc", json=body).status_code == 200
        assert c.put("/sessions/bad id!", json=body).status_code == 400
        assert [s["id"] for s in c.get("/sessions").json()] == ["abc"]
        assert c.delete("/sessions/abc").json() == {"ok": True}
        assert c.get("/sessions").json() == []
