# repochat

Chat with any public GitHub repository. Paste a repo URL, the app indexes it, then a LangGraph agent answers questions with `path:line` citations.

Live: https://repochat.samabiodun.tech

## How it works

1. **Ingest.** `POST /repos` clones the repo (depth 1), skips binaries and vendored folders, caps the total at 4 MB of text, and splits every file into 60-line chunks with a 10-line overlap.
2. **Index.** Chunks are embedded with `text-embedding-3-small` and stored in Azure AI Search with their file path and line range. Full files go into a second index for exact lookups.
3. **Agent.** `POST /chat` runs a LangGraph ReAct loop over `gpt-4.1-mini` with four tools: hybrid search (vector + keyword), grep, read file, and list files. The model is told to search first, retry with different words when results look thin, cite `path:line`, and say so when the repo has no answer. A recursion cap stops runaway loops.
4. **Stream.** Tokens and tool calls stream back over Server-Sent Events. The single-page UI at `/` shows the tool calls as they happen.

## Stack

Python 3.12, FastAPI, LangGraph, LangChain, Azure OpenAI (via Azure AI Foundry), Azure AI Search, Docker, GitHub Actions, Azure Container Apps.

## Run locally

```bash
cp .env.example .env   # fill in the Azure values
uv sync
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000.

## Tests and eval

```bash
uv run pytest
uv run python eval/run_eval.py http://localhost:8000
```

`eval/golden.jsonl` holds questions about `psf/requests` with expected substrings. The eval indexes the repo, asks each question through the API, and reports pass/fail. One question has no answer in the repo, to check the agent admits that instead of inventing one.

## Deploy

Every push to `main` runs lint and tests, builds the image, pushes it to GHCR, and deploys it to Azure Container Apps. Secrets live in GitHub Actions and the Container App, never in the repo.

## Limits

Public repos only. Repos over the text budget are indexed partially and the skipped files are listed in the job status. The free Azure AI Search tier holds about 50 MB, so a handful of repos at a time.

## Sign-in and saved sessions

Sessions save in the browser by default. Signing in with GitHub (OAuth, user identity only) moves them to Cosmos DB so they follow you across devices. Routes: `/auth/login`, `/auth/callback`, `/auth/me`, `/auth/logout`, and `GET/PUT/DELETE /sessions`. Needs `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, `SESSION_SECRET`, `COSMOS_ENDPOINT`, `COSMOS_KEY`; without them the sign-in link is hidden and sessions stay local.
