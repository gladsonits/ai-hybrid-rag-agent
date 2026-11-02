import asyncio
import os
from unittest.mock import MagicMock
from augmentation.chat_service import ChatService
from augmentation.augmenter import Augmenter
from generation.responder import Responder
from augmentation.session_store import SessionStore
from memory.cache import RedisAnswerCache
from memory.faq import PostgresFaqMatcher
from memory.faq import FaqMetrics
from memory.config import FaqSettings
from retrieval.database import Database
from retrieval.embedder import EmbeddingModel
from retrieval.vector_store import VectorStore
from retrieval.text_search import TextSearchStore
from retrieval.retriever import HybridRetriever
from generation.llm import LlmClient

class MockLlmClient(LlmClient):
    async def generate(self, messages, **kwargs):
        return "Mocked Answer"

class MockResponder(Responder):
    async def respond(self, messages, passages):
        yield {"type": "token", "token": "Mocked "}
        yield {"type": "token", "token": "Answer"}
        yield {"type": "done", "answer": "Mocked Answer", "citations": []}

async def main():
    # Setup
    db = Database()
    embedder = EmbeddingModel()
    
    # Mock the LLM and Responder to avoid API keys
    llm = MockLlmClient()
    responder = MockResponder()
    
    # Setup real memory and retrieval
    cache = RedisAnswerCache()
    faq_matcher = PostgresFaqMatcher(db, embedder)
    faq_matcher.ensure_schema()
    
    # Setup other services
    session_store = SessionStore(db)
    retriever = HybridRetriever(db, embedder)
    augmenter = Augmenter(retriever, llm)
    
    service = ChatService(
        augmenter=augmenter,
        responder=responder,
        session_store=session_store,
        answer_cache=cache,
        faq_matcher=faq_matcher,
        faq_metrics=FaqMetrics(db),
        index_state=db # simplified
    )

    # Test cases
    q1 = "What is Rose's current project?"
    q2 = "What is Rose's current project?" # Exact
    q3 = "What is the current project of Rose?" # Semantic/Keyword

    print("--- Turn 1: Initial Question ---")
    async for event in service.stream_answer("s1", q1):
        print(event)

    print("\n--- Turn 2: Exact Same Question ---")
    async for event in service.stream_answer("s1", q2):
        print(event)

    print("\n--- Turn 3: Slightly Different Question ---")
    async for event in service.stream_answer("s1", q3):
        print(event)

if __name__ == "__main__":
    asyncio.run(main())
