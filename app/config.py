from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = "gpt-4.1-mini"
    azure_openai_embed_deployment: str = "text-embedding-3-small"
    azure_search_endpoint: str
    azure_search_api_key: str
    max_repo_bytes: int = 4_000_000
    embed_dimensions: int = 1536
    github_client_id: str = ""
    github_client_secret: str = ""
    session_secret: str = "dev-only-change-me"
    cosmos_endpoint: str = ""
    cosmos_key: str = ""


@lru_cache
def settings() -> Settings:
    return Settings()
