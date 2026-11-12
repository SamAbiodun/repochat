import time
from functools import lru_cache

from azure.cosmos import ContainerProxy, CosmosClient, PartitionKey, exceptions

from app.config import settings

MAX_SESSION_BYTES = 200_000
MAX_SESSIONS = 100


@lru_cache
def container() -> ContainerProxy:
    cfg = settings()
    client = CosmosClient(cfg.cosmos_endpoint, cfg.cosmos_key)
    db = client.create_database_if_not_exists("repochat")
    return db.create_container_if_not_exists("sessions", partition_key=PartitionKey("/userId"))


def list_sessions(user_id: str) -> list[dict]:
    rows = container().query_items(
        "SELECT c.id, c.repo, c.title, c.updated FROM c WHERE c.userId = @u "
        "ORDER BY c.updated DESC OFFSET 0 LIMIT @n",
        parameters=[{"name": "@u", "value": user_id}, {"name": "@n", "value": MAX_SESSIONS}],
        partition_key=user_id,
    )
    return list(rows)


def get_session(user_id: str, session_id: str) -> dict | None:
    try:
        doc = container().read_item(session_id, partition_key=user_id)
    except exceptions.CosmosResourceNotFoundError:
        return None
    return {k: doc[k] for k in ("id", "repo", "title", "updated", "messages")}


def put_session(user_id: str, session_id: str, repo: str, title: str, messages: list) -> dict:
    doc = {
        "id": session_id,
        "userId": user_id,
        "repo": repo,
        "title": title[:120],
        "updated": int(time.time() * 1000),
        "messages": messages,
    }
    container().upsert_item(doc)
    return {k: doc[k] for k in ("id", "repo", "title", "updated")}


def delete_session(user_id: str, session_id: str) -> None:
    try:
        container().delete_item(session_id, partition_key=user_id)
    except exceptions.CosmosResourceNotFoundError:
        pass
