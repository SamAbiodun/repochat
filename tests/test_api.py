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
