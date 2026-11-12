import json
import sys
import time
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
cases = [
    json.loads(line)
    for line in (Path(__file__).parent / "golden.jsonl").read_text().splitlines()
    if line
]


def ensure_indexed(client: httpx.Client, url: str) -> str:
    repo = client.post("/repos", json={"url": url}).json()["repo"]
    while True:
        job = client.get(f"/repos/{repo}").json()
        if job["status"] == "ready":
            return repo
        if job["status"] == "failed":
            raise SystemExit(f"index failed: {job['error']}")
        time.sleep(2)


def ask(client: httpx.Client, repo: str, question: str) -> str:
    answer = ""
    with client.stream(
        "POST", "/chat", json={"repo": repo, "messages": [{"role": "user", "content": question}]}
    ) as r:
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event == "token":
                answer += json.loads(line[6:])
    return answer


passed = 0
with httpx.Client(base_url=BASE, timeout=600) as client:
    for case in cases:
        repo = ensure_indexed(client, case["repo"])
        answer = ask(client, repo, case["question"])
        ok = any(e.lower() in answer.lower() for e in case["expect"])
        passed += ok
        print(
            f"{'PASS' if ok else 'FAIL'}  {case['question']}\n      {answer[:200].replace(chr(10), ' ')}\n"
        )
print(f"{passed}/{len(cases)} passed")
sys.exit(0 if passed == len(cases) else 1)
