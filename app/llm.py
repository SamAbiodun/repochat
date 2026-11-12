from functools import lru_cache

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from app.config import settings


@lru_cache
def chat_model() -> AzureChatOpenAI:
    s = settings()
    return AzureChatOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,
        api_version=s.azure_openai_api_version,
        azure_deployment=s.azure_openai_chat_deployment,
        temperature=0,
        streaming=True,
    )


@lru_cache
def embeddings() -> AzureOpenAIEmbeddings:
    s = settings()
    return AzureOpenAIEmbeddings(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,
        api_version=s.azure_openai_api_version,
        azure_deployment=s.azure_openai_embed_deployment,
    )
