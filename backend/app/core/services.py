"""Central service instances for TalentScreen (PDF Part 5, 7, 9).

Initializes and exposes global, reusable service singletons:
  - VectorDB: Milvus dense vector storage management
  - EmbeddingService: all-MiniLM-L6-v2 embedding generation
  - HybridRetrievalService: BM25 keyword + Milvus vector retrieval fusion
  - LLMService: Configured local or cloud LLM helper actions
"""

from app.db.vector_db import VectorDB
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import HybridRetrievalService
from app.services.llm_service import LLMService

# Global service singletons
vector_db = VectorDB()
embedding_service = EmbeddingService()
retrieval_service = HybridRetrievalService(vector_db, embedding_service)
llm_service = LLMService()
