import os

import pytest

from app.ingest import chunk_file, collect_files, parse_repo


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/tiangolo/fastapi",
        "github.com/tiangolo/fastapi/",
        "https://github.com/tiangolo/fastapi.git",
        "https://www.github.com/tiangolo/FastAPI",
    ],
)
def test_parse_repo(url):
    assert parse_repo(url) == "tiangolo/fastapi"


@pytest.mark.parametrize("url", ["https://gitlab.com/a/b", "tiangolo", "https://github.com/a"])
def test_parse_repo_rejects(url):
    with pytest.raises(ValueError):
        parse_repo(url)


def test_chunk_file_tracks_lines():
    text = "\n".join(f"line {i}" for i in range(1, 131))
    chunks = chunk_file("x.py", text)
    assert [(c["start_line"], c["end_line"]) for c in chunks] == [(1, 60), (51, 110), (101, 130)]
    assert chunks[0]["content"].startswith("# x.py\nline 1\n")
    assert chunk_file("empty.py", "\n\n") == []


def test_collect_files_skips_binaries_and_budget(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\n")
    (tmp_path / "b.png").write_bytes(b"\x89PNG")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01")
    os.makedirs(tmp_path / "node_modules")
    (tmp_path / "node_modules" / "d.js").write_text("x")
    (tmp_path / "big.txt").write_text("y" * 50)
    kept, skipped = collect_files(str(tmp_path), budget=20)
    assert [p for p, _ in kept] == ["a.py"]
    assert skipped == ["big.txt"]
