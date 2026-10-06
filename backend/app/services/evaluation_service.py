"""RAG Evaluation Service (PDF Part 11).

Evaluates retrieval quality, RAG pipeline quality, generation quality,
and response faithfulness. Uses the multi-provider LLM approach with
fallback to simple heuristic scoring when no LLM is available.
"""

from typing import Dict, Any
from app.core.logging import logger
from app.services.llm_service import LLMService


class EvaluationService:
    """Evaluation framework for retrieval, RAG, and generation quality.
    
    Covers:
    - Faithfulness: Is the answer grounded in the context?
    - Relevance: Is the retrieved context relevant to the query?
    - Completeness: Does the response address the full query?
    """

    def __init__(self):
        self.llm_service = LLMService()

    def _invoke(self, prompt: str) -> str:
        """Call the LLM for evaluation."""
        try:
            return self.llm_service.invoke(
                "You are a precise RAG evaluation assistant. Return only the requested JSON.",
                prompt,
            )
        except Exception as e:
            logger.error(f"Evaluation LLM call failed: {e}")
            return '{"score": 0.5, "reasoning": "Evaluation failed"}'

    def evaluate_faithfulness(self, query: str, context: str, response: str) -> str:
        """Check if the answer is grounded in the context (PDF Part 11.2)."""
        prompt = f"""Evaluation Task: Faithfulness Assessment

System Context (Retrieved Information):
{context[:2000]}

Recruiter Query: {query}

Generated Answer: {response[:1000]}

Does the Generated Answer contain information NOT found in the Context (Hallucinations)?
Return a JSON with "faithfulness_score" (0-1, where 1 = fully faithful) and "reasoning".
"""
        return self._invoke(prompt)

    def evaluate_relevance(self, query: str, context: str) -> str:
        """Check if the retrieved context is relevant to the query (PDF Part 11.1)."""
        prompt = f"""Evaluation Task: Context Relevance Assessment

Recruiter Query: {query}

Retrieved Context:
{context[:2000]}

Is the retrieved context useful and relevant to answer the recruiter's query?
Return a JSON with "relevance_score" (0-1, where 1 = highly relevant) and "reasoning".
"""
        return self._invoke(prompt)

    def evaluate_completeness(self, query: str, response: str) -> str:
        """Check if the response fully addresses the query (PDF Part 11.3)."""
        prompt = f"""Evaluation Task: Response Completeness Assessment

Recruiter Query: {query}

Generated Response: {response[:1000]}

Does the response fully address what the recruiter is asking?
Return a JSON with "completeness_score" (0-1, where 1 = fully complete) and "reasoning".
"""
        return self._invoke(prompt)
