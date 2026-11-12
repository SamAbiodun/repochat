import asyncio
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from app import search
from app.config import settings
from app.llm import embeddings

SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "target",
    "vendor",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".idea",
    "coverage",
    ".tox",
    "site-packages",
}
SKIP_EXT = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".webp",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".mp3",
    ".mp4",
    ".wav",
    ".lock",
    ".min.js",
    ".map",
    ".pyc",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".parquet",
    ".db",
    ".sqlite",
}
MAX_FILE_BYTES = 200_000
CHUNK_LINES = 60
OVERLAP_LINES = 10

GITHUB_RE = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


@dataclass
class Job:
    repo: str
    status: str = "queued"
    files: int = 0
    chunks: int = 0
    error: str | None = None
    skipped: list[str] = field(default_factory=list)


JOBS: dict[str, Job] = {}


def parse_repo(url: str) -> str:
    m = GITHUB_RE.match(url.strip())
    if not m:
        raise ValueError("Expected a public GitHub URL like https://github.com/owner/repo")
    return f"{m.group(1)}/{m.group(2)}".lower()


def _is_text(data: bytes) -> bool:
    return b"\x00" not in data[:8000]


def collect_files(root: str, budget: int) -> tuple[list[tuple[str, str]], list[str]]:
    kept: list[tuple[str, str]] = []
    skipped: list[str] = []
    used = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            lower = name.lower()
            if any(lower.endswith(ext) for ext in SKIP_EXT):
                continue
            full = os.path.join(dirpath, name)
            size = os.path.getsize(full)
            if size > MAX_FILE_BYTES or size == 0:
                skipped.append(rel)
                continue
            if used + size > budget:
                skipped.append(rel)
                continue
            with open(full, "rb") as f:
                data = f.read()
            if not _is_text(data):
                continue
            kept.append((rel, data.decode("utf-8", errors="replace")))
            used += size
    return kept, skipped


def chunk_file(path: str, text: str) -> list[dict]:
    lines = text.splitlines()
    chunks: list[dict] = []
    start = 0
    while start < len(lines):
        end = min(start + CHUNK_LINES, len(lines))
        body = "\n".join(lines[start:end])
        if body.strip():
            chunks.append(
                {
                    "path": path,
                    "start_line": start + 1,
                    "end_line": end,
                    "content": f"# {path}\n{body}",
                }
            )
        if end == len(lines):
            break
        start = end - OVERLAP_LINES
    return chunks


def clone(repo: str, dest: str) -> None:
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", f"https://github.com/{repo}.git", dest],
        check=True,
        capture_output=True,
        timeout=120,
    )


def _doc_id(repo: str, path: str, suffix: str = "") -> str:
    return hashlib.sha1(f"{repo}:{path}:{suffix}".encode()).hexdigest()


def index_repo(repo: str) -> Job:
    job = JOBS[repo]
    job.status = "cloning"
    tmp = tempfile.mkdtemp(prefix="repochat-")
    try:
        clone(repo, tmp)
        job.status = "indexing"
        files, job.skipped = collect_files(tmp, settings().max_repo_bytes)
        job.files = len(files)
        search.delete_repo(repo)
        search.upload(
            search.FILES_INDEX,
            [{"id": _doc_id(repo, p), "repo": repo, "path": p, "content": t} for p, t in files],
        )
        chunks = [c for p, t in files for c in chunk_file(p, t)]
        emb = embeddings()
        for i in range(0, len(chunks), 64):
            batch = chunks[i : i + 64]
            vectors = emb.embed_documents([c["content"] for c in batch])
            search.upload(
                search.CHUNKS_INDEX,
                [
                    {
                        "id": _doc_id(repo, c["path"], str(c["start_line"])),
                        "repo": repo,
                        "vector": v,
                        **c,
                    }
                    for c, v in zip(batch, vectors)
                ],
            )
            job.chunks = i + len(batch)
        job.status = "ready"
    except subprocess.CalledProcessError as e:
        job.status = "failed"
        job.error = "git clone failed: " + e.stderr.decode(errors="replace").strip()[-300:]
    except Exception as e:
        job.status = "failed"
        job.error = f"{type(e).__name__}: {e}"[:300]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return job


async def start_index(repo: str) -> Job:
    existing = JOBS.get(repo)
    if existing and existing.status in {"queued", "cloning", "indexing", "ready"}:
        return existing
    if search.repo_indexed(repo):
        JOBS[repo] = Job(repo=repo, status="ready")
        return JOBS[repo]
    JOBS[repo] = Job(repo=repo)
    asyncio.get_running_loop().run_in_executor(None, index_repo, repo)
    return JOBS[repo]
