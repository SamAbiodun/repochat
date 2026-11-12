import re
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from app.config import settings

CHUNKS_INDEX = "repochat-chunks"
FILES_INDEX = "repochat-files"
PROFILE = "hnsw"


def _index_client() -> SearchIndexClient:
    s = settings()
    return SearchIndexClient(s.azure_search_endpoint, AzureKeyCredential(s.azure_search_api_key))


@lru_cache
def client(index: str) -> SearchClient:
    s = settings()
    return SearchClient(s.azure_search_endpoint, index, AzureKeyCredential(s.azure_search_api_key))


def _text(name: str) -> SearchField:
    return SearchField(
        name=name,
        type=SearchFieldDataType.String,
        searchable=True,
        filterable=False,
        sortable=False,
        facetable=False,
    )


def ensure_indexes() -> None:
    dims = settings().embed_dimensions
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[VectorSearchProfile(name=PROFILE, algorithm_configuration_name="hnsw-algo")],
    )
    chunks = SearchIndex(
        name=CHUNKS_INDEX,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="repo", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="path", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="start_line", type=SearchFieldDataType.Int32),
            SimpleField(name="end_line", type=SearchFieldDataType.Int32),
            _text("content"),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dims,
                vector_search_profile_name=PROFILE,
            ),
        ],
        vector_search=vector_search,
    )
    files = SearchIndex(
        name=FILES_INDEX,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="repo", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="path", type=SearchFieldDataType.String, searchable=True, filterable=True
            ),
            _text("content"),
        ],
    )
    ic = _index_client()
    ic.create_or_update_index(chunks)
    ic.create_or_update_index(files)


def safe_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-=]", "_", text)


def delete_repo(repo: str) -> None:
    for index in (CHUNKS_INDEX, FILES_INDEX):
        c = client(index)
        ids = [
            {"id": d["id"]}
            for d in c.search("*", filter=f"repo eq '{repo}'", select=["id"], top=100000)
        ]
        for i in range(0, len(ids), 1000):
            c.delete_documents(ids[i : i + 1000])


def upload(index: str, docs: list[dict]) -> None:
    c = client(index)
    for i in range(0, len(docs), 500):
        failed = [r for r in c.upload_documents(docs[i : i + 500]) if not r.succeeded]
        if failed:
            raise RuntimeError(f"{len(failed)} documents rejected: {failed[0].error_message}")


def hybrid_search(repo: str, query: str, vector: list[float], k: int = 8) -> list[dict]:
    results = client(CHUNKS_INDEX).search(
        search_text=query,
        vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields="vector")],
        filter=f"repo eq '{repo}'",
        select=["path", "start_line", "end_line", "content"],
        top=k,
    )
    return [dict(r) for r in results]


def get_file(repo: str, path: str) -> str | None:
    results = client(FILES_INDEX).search(
        search_text="*",
        filter=f"repo eq '{repo}' and path eq '{path.replace(chr(39), chr(39) * 2)}'",
        select=["content"],
        top=1,
    )
    for r in results:
        return r["content"]
    return None


def list_paths(repo: str) -> list[str]:
    results = client(FILES_INDEX).search(
        search_text="*", filter=f"repo eq '{repo}'", select=["path"], top=5000
    )
    return sorted(r["path"] for r in results)


def grep(repo: str, pattern: str, max_hits: int = 30) -> list[dict]:
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    results = client(FILES_INDEX).search(
        search_text=pattern,
        filter=f"repo eq '{repo}'",
        select=["path", "content"],
        top=50,
        query_type="simple",
        search_mode="any",
    )
    hits: list[dict] = []
    for r in results:
        for n, line in enumerate(r["content"].splitlines(), start=1):
            if rx.search(line):
                hits.append({"path": r["path"], "line": n, "text": line.strip()[:200]})
                if len(hits) >= max_hits:
                    return hits
    return hits


def repo_indexed(repo: str) -> bool:
    results = client(FILES_INDEX).search("*", filter=f"repo eq '{repo}'", select=["id"], top=1)
    return any(True for _ in results)
