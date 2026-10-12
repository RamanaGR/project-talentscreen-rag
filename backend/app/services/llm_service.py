"""LLM Service (PDF Parts 6, 9).

Multi-provider LLM service with fallback chain:
1. Ollama (local development)
2. AWS Bedrock (Claude via Amazon Bedrock — production)
3. Anthropic Direct API (Claude)
4. OpenAI (GPT-4)
5. G4F or mock responses (offline development)

Provides:
- Query Understanding (Part 6.1): Extract skills, roles, experience, intent
- Query Optimization (Part 6.2): Remove noise, normalize terminology
- Query Expansion (Part 6.3): Add synonyms and related terms
- Context Building (Part 9.2): Prepare context for generation
- Response Generation (Part 9.3): Stream responses from Claude
- Prompt Engineering (Part 9.1): Hallucination reduction guardrails
"""

import json
import re
from typing import List, Dict, Any, Generator
import httpx
from app.core.config import settings
from app.core.logging import logger

# --- Optional imports ---
try:
    import anthropic as anthropic_sdk
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import g4f
    G4F_AVAILABLE = True
except ImportError:
    G4F_AVAILABLE = False


class LLMService:
    """Provider-neutral text generation for search, ingestion, and chat."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.strip().lower()
        self.use_bedrock = False
        self.bedrock_client = None
        self.anthropic_client = None
        self.openai_llm = None
        self.use_ollama = False
        self.use_g4f = False
        self._init_clients()

    def _init_clients(self):
        # Keep local Ollama isolated from cloud credentials and clients.
        if self.provider == "ollama":
            self.use_ollama = True
            logger.info(
                f"[LLM] Configured local Ollama: {settings.OLLAMA_BASE_URL} "
                f"(model={settings.OLLAMA_MODEL})"
            )
            return

        # 1. AWS Bedrock (Production)
        if self.provider == "bedrock" and BOTO3_AVAILABLE:
            aws_id = settings.AWS_ACCESS_KEY_ID
            aws_secret = settings.AWS_SECRET_ACCESS_KEY
            if aws_id and aws_secret and "your_aws" not in aws_id.lower():
                try:
                    self.bedrock_client = boto3.client(
                        service_name='bedrock-runtime',
                        region_name=settings.AWS_REGION,
                        aws_access_key_id=aws_id,
                        aws_secret_access_key=aws_secret
                    )
                    self.use_bedrock = True
                    logger.info(f"[LLM] Initialized Amazon Bedrock: {settings.BEDROCK_MODEL_ID}")
                except Exception as e:
                    logger.error(f"[LLM] Bedrock init failed: {e}")
            else:
                logger.info("[LLM] AWS Bedrock credentials not configured. Skipping.")
        elif self.provider == "bedrock":
            logger.info("[LLM] boto3 not installed. Bedrock unavailable.")

        # 2. Anthropic Direct API (Local Dev — uses key from .env)
        anthropic_key = settings.ANTHROPIC_API_KEY
        if self.provider == "anthropic" and ANTHROPIC_AVAILABLE and anthropic_key and "sk-ant" in anthropic_key:
            try:
                self.anthropic_client = anthropic_sdk.Anthropic(api_key=anthropic_key)
                logger.info("[LLM] Initialized Anthropic Claude (direct API).")
            except Exception as e:
                logger.error(f"[LLM] Anthropic init failed: {e}")
        else:
            if self.provider == "anthropic" and not ANTHROPIC_AVAILABLE:
                logger.info("[LLM] 'anthropic' package not installed. Skipping.")
            elif self.provider == "anthropic":
                logger.info("[LLM] Anthropic API key not configured. Skipping.")

        # 3. OpenAI (Secondary fallback)
        openai_key = settings.OPENAI_API_KEY
        if self.provider == "openai" and OPENAI_AVAILABLE and openai_key and "your_openai" not in openai_key.lower():
            try:
                self.openai_llm = ChatOpenAI(
                    model_name="gpt-4-turbo",
                    api_key=openai_key,
                    temperature=0.2
                )
                logger.info("[LLM] Initialized OpenAI GPT-4.")
            except Exception as e:
                logger.error(f"[LLM] OpenAI init failed: {e}")
        # 4. G4F (GPT4Free) Fallback
        if self.provider == "g4f" and G4F_AVAILABLE:
            self.use_g4f = True
            logger.info("[LLM] Initialized G4F (GPT4Free) for free API access.")
        elif self.provider == "g4f":
            self.use_g4f = False
            logger.info("[LLM] G4F is unavailable. Using mock fallback.")
        elif self.provider not in {"bedrock", "anthropic", "openai"}:
            logger.warning(f"[LLM] Unknown provider '{self.provider}'. Using mock fallback.")

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """Generate one response using the configured provider.

        A selected provider never falls through to a different remote provider.
        """
        if self.use_ollama:
            try:
                return self._invoke_ollama(system_prompt, user_prompt)
            except Exception as e:
                logger.error(f"[LLM] Ollama invocation failed: {e}. Using mock fallback.")
                return self._get_mock_response(user_prompt, system_prompt)

        # 1. AWS Bedrock
        if self.use_bedrock and self.bedrock_client:
            try:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "temperature": 0.2
                })
                response = self.bedrock_client.invoke_model(
                    modelId=settings.BEDROCK_MODEL_ID,
                    body=body
                )
                response_body = json.loads(response.get('body').read())
                return response_body.get('content', [{}])[0].get('text', '')
            except Exception as e:
                logger.error(f"[LLM] Bedrock invocation failed: {e}. Trying Anthropic...")

        # 2. Anthropic Direct
        if self.anthropic_client:
            try:
                message = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.2
                )
                return message.content[0].text
            except Exception as e:
                logger.error(f"[LLM] Anthropic invocation failed: {e}. Trying OpenAI...")

        # 3. OpenAI
        if self.openai_llm:
            try:
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
                res = self.openai_llm.invoke(messages)
                return res.content
            except Exception as e:
                logger.error(f"[LLM] OpenAI invocation failed: {e}. Trying G4F...")

        # 4. G4F (GPT4Free) Free Tier
        if self.use_g4f:
            try:
                response = g4f.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response
            except Exception as e:
                logger.error(f"[LLM] G4F invocation failed: {e}")

        # 5. Mock fallback
        return self._get_mock_response(user_prompt, system_prompt)

    def _invoke_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama's non-streaming local chat endpoint."""
        response = httpx.post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama response did not contain message.content")
        return content

    # ================================================================
    # QUERY PROCESSING (PDF Part 6)
    # ================================================================

    def understand_query(self, query: str) -> Dict[str, Any]:
        """Query Understanding (PDF Part 6.1).
        
        Analyzes recruiter query to identify:
        - Required Skills
        - Job Roles
        - Experience Requirements
        - Search Intent (candidate_search, knowledge_retrieval, comparison, summarization)
        """
        system_prompt = (
            "You are a Senior Technical Recruiter AI. Analyze the recruiter's search query "
            "and extract structured information.\n"
            "Return ONLY a valid JSON object with these fields:\n"
            '- "skills": list of skills/technologies mentioned\n'
            '- "roles": list of job roles mentioned\n'
            '- "experience_years": minimum years of experience required (0 if not specified)\n'
            '- "intent": one of "candidate_search", "knowledge_retrieval", "comparison", "summarization"\n'
            '- "certifications": list of certifications mentioned\n'
            '- "other_requirements": any other requirements as a list of strings'
        )
        user_prompt = f"Recruiter Query: {query}"
        
        try:
            response = self.invoke(system_prompt, user_prompt)
            # Extract JSON from response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"Query understanding failed: {e}")
        
        # Fallback: simple keyword extraction
        return self._fallback_understand_query(query)

    def _fallback_understand_query(self, query: str) -> Dict[str, Any]:
        """Simple regex-based query understanding fallback."""
        query_lower = query.lower()
        
        # Detect intent
        intent = "candidate_search"
        if any(w in query_lower for w in ["compare", "versus", "vs", "difference"]):
            intent = "comparison"
        elif any(w in query_lower for w in ["summarize", "summary", "overview"]):
            intent = "summarization"
        elif any(w in query_lower for w in ["guideline", "policy", "process", "procedure", "interview"]):
            intent = "knowledge_retrieval"
        
        # Extract experience years
        exp_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)', query, re.IGNORECASE)
        experience = int(exp_match.group(1)) if exp_match else 0
        
        return {
            "skills": [],
            "roles": [],
            "experience_years": experience,
            "intent": intent,
            "certifications": [],
            "other_requirements": []
        }

    def optimize_query(self, query: str) -> str:
        """Query Optimization (PDF Part 6.2).
        
        Makes the query retrieval-friendly:
        - Removes unnecessary text
        - Normalizes terminology
        - Standardizes role names
        """
        system_prompt = (
            "You are a query optimizer for a recruitment search system. "
            "Clean and optimize the recruiter's query for better search results.\n"
            "Rules:\n"
            "- Remove conversational filler words (can you, please, some, good)\n"
            "- Keep all technical terms, skills, roles, and requirements\n"
            "- Normalize role names (e.g., 'dev' → 'Developer')\n"
            "- Keep experience requirements\n"
            "- Return ONLY the optimized query string, nothing else"
        )
        user_prompt = f"Original Query: {query}"
        
        try:
            optimized = self.invoke(system_prompt, user_prompt).strip()
            optimized = optimized.replace('"', '').replace("'", "")
            return optimized if optimized and len(optimized) > 3 else query
        except Exception as e:
            logger.warning(f"Query optimization failed: {e}")
            return query

    def expand_query(self, query: str) -> str:
        """Query Expansion (PDF Part 6.3).
        
        Enriches the query with semantically related terms:
        - Alternate job titles
        - Related technologies
        - Skill synonyms
        """
        system_prompt = (
            "You are a Senior Technical Recruiter. Expand candidate search queries for deep semantic vector search. "
            "Identify synonyms, related technologies, alternate job titles, and seniority tags.\n"
            "For example:\n"
            "- 'Machine Learning Engineer' → 'Machine Learning Engineer, AI Engineer, NLP Engineer, Deep Learning Engineer, ML Developer'\n"
            "- 'Data Engineer' → 'Data Engineer, Data Platform Engineer, Big Data Engineer, Analytics Engineer, ETL Developer'"
        )
        user_prompt = (
            f"Original Search Query: '{query}'\n\n"
            "Respond only with the expanded comma-separated keywords and concepts. Do not explain."
        )
        try:
            expanded = self.invoke(system_prompt, user_prompt).strip()
            expanded = expanded.replace('"', '').replace("'", "")
            return expanded if expanded else query
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query

    # ================================================================
    # RESPONSE GENERATION (PDF Part 9)
    # ================================================================

    def stream_response(self, query: str, context_documents: List[Dict[str, Any]]) -> Generator[Any, None, None]:
        """Stream a response from the LLM using retrieved context (PDF Part 9.3).
        
        Uses prompt engineering to ensure Claude:
        - Uses ONLY retrieved information
        - Generates recruiter-friendly responses
        - Avoids unsupported assumptions
        - Provides clear explanations
        - Maintains consistent formatting
        """
        context_text = self._build_context(context_documents)

        system_prompt = (
            "You are TalentScreen, an expert AI Recruitment Knowledge Assistant. "
            "Your goal is to help recruiters evaluate candidate profiles and access recruitment knowledge.\n\n"
            "CRITICAL RULES:\n"
            "1. Answer ONLY using the information provided in the CONTEXT below.\n"
            "2. DO NOT generate or assume information that is not present in the context.\n"
            "3. If the context does not contain enough information to answer, clearly state that.\n"
            "4. When recommending candidates, explain WHY they match the requirements.\n"
            "5. Highlight specific skills, experience, and qualifications from the context.\n"
            "6. Use professional, recruiter-friendly language.\n"
            "7. Structure your response clearly with sections when appropriate.\n"
            "8. If comparing candidates, use a structured format with pros/cons.\n"
            "9. Never fabricate skills, experience, or certifications not in the source documents.\n"
            "10. Provide evidence from the resumes/documents to back up every claim."
        )
        user_prompt = f"Recruiter Query: {query}\n\nCONTEXT:\n{context_text}"

        # Ollama streams newline-delimited JSON from its local HTTP endpoint.
        if self.use_ollama:
            try:
                yield from self._stream_ollama(system_prompt, user_prompt)
            except Exception as e:
                logger.error(f"[LLM] Ollama streaming failed: {e}. Using mock fallback.")
                yield from self._stream_mock(query)
            return

        # 1. AWS Bedrock Streaming
        if self.use_bedrock and self.bedrock_client:
            try:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "temperature": 0.2
                })
                response = self.bedrock_client.invoke_model_with_response_stream(
                    modelId=settings.BEDROCK_MODEL_ID,
                    body=body
                )
                stream = response.get('body')
                if stream:
                    for event in stream:
                        chunk = event.get('chunk')
                        if chunk:
                            chunk_data = json.loads(chunk.get('bytes').decode('utf-8'))
                            if chunk_data.get('type') == 'content_block_delta':
                                text = chunk_data.get('delta', {}).get('text', '')
                                yield _TextChunk(text)
                    return
            except Exception as e:
                logger.error(f"[LLM] Bedrock streaming failed: {e}. Trying Anthropic...")

        # 2. Anthropic Direct Streaming
        if self.anthropic_client:
            try:
                with self.anthropic_client.messages.stream(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.2
                ) as stream:
                    for text in stream.text_stream:
                        yield _TextChunk(text)
                return
            except Exception as e:
                logger.error(f"[LLM] Anthropic streaming failed: {e}. Trying OpenAI...")

        # 3. OpenAI Streaming
        if self.openai_llm:
            try:
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
                for chunk in self.openai_llm.stream(messages):
                    yield chunk
                return
            except Exception as e:
                logger.error(f"[LLM] OpenAI streaming failed: {e}")

        # 4. G4F Streaming
        if self.use_g4f:
            try:
                response = g4f.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=True
                )
                for chunk in response:
                    if isinstance(chunk, str):
                        yield _TextChunk(chunk)
                return
            except Exception as e:
                logger.error(f"[LLM] G4F streaming failed: {e}")

        # 5. Mock Streaming Fallback
        mock_response = self._get_mock_response(query)
        words = mock_response.split(" ")
        for i in range(0, len(words), 3):
            word_chunk = " ".join(words[i:i+3]) + " "
            yield _TextChunk(word_chunk)

    def _stream_ollama(self, system_prompt: str, user_prompt: str) -> Generator[Any, None, None]:
        """Translate Ollama JSONL stream events into application text chunks."""
        with httpx.stream(
            "POST",
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": True,
                "options": {"temperature": 0.2},
            },
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                content = payload.get("message", {}).get("content", "")
                if content:
                    yield _TextChunk(content)

    def _stream_mock(self, query: str) -> Generator[Any, None, None]:
        """Yield the existing offline response in small chunks."""
        words = self._get_mock_response(query).split(" ")
        for i in range(0, len(words), 3):
            yield _TextChunk(" ".join(words[i:i + 3]) + " ")

    def _build_context(self, context_documents: List[Dict[str, Any]]) -> str:
        """Context Building (PDF Part 9.2).
        
        Organizes retrieved information into structured context:
        - Selects top-ranked chunks
        - Removes duplicate information
        - Prepares structured context for Claude
        """
        if not context_documents:
            return "No relevant documents found."
        
        seen_texts = set()
        context_parts = []
        
        for i, doc in enumerate(context_documents, 1):
            text = doc.get('text', '')
            # Skip duplicates
            text_hash = text[:200]
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)
            
            metadata = doc.get('metadata', {})
            doc_type = metadata.get('document_type', 'document')
            filename = metadata.get('filename', 'Unknown')
            
            header = f"--- Document {i} [{doc_type.upper()}] ({filename}) ---"
            context_parts.append(f"{header}\n{text}")
        
        return "\n\n".join(context_parts)

    def _get_mock_response(self, user_prompt: str, system_prompt: str = "") -> str:
        """Mock response for offline development without LLM access."""
        system_lower = system_prompt.lower()
        if "expand candidate search queries" in system_lower:
            return "software engineering, cloud architecture, system design"
            
        if "optimizer for a recruitment search system" in system_lower:
            return user_prompt.replace("Original Query: ", "").strip()
            
        if "valid json object" in system_lower:
            return '{"skills": [], "roles": [], "experience_years": 0, "intent": "candidate_search", "certifications": [], "other_requirements": []}'

        user_prompt_lower = user_prompt.lower()
        if "snowflake" in user_prompt_lower or "data engineer" in user_prompt_lower:
            return (
                "## Candidate Analysis\n\n"
                "Based on the retrieved documents, here are the matching candidates:\n\n"
                "### Jane Doe — Senior Data Engineer\n"
                "- **Experience**: 7 years in Data Engineering\n"
                "- **Key Skills**: Snowflake, AWS, Python, Airflow, dbt\n"
                "- **Certifications**: SnowPro Core, AWS Solutions Architect\n"
                "- **Recommendation**: Highly recommended — extensive Snowflake experience with enterprise data lake projects.\n\n"
                "### John Smith — Data Engineer\n"
                "- **Experience**: 4 years in Data Engineering\n"
                "- **Key Skills**: Snowflake, dbt, SQL, GCP\n"
                "- **Recommendation**: Good match — solid Snowflake skills with growing experience.\n\n"
                "**Why Jane Doe is the stronger match**: She has 3 more years of experience and holds a SnowPro Core certification."
            )
        if "compare" in user_prompt_lower:
            return (
                "## Candidate Comparison\n\n"
                "| Criteria | Candidate A | Candidate B |\n"
                "|----------|------------|------------|\n"
                "| Experience | 5 years | 3 years |\n"
                "| Key Skills | Python, AWS, ML | Java, Azure, Data |\n"
                "| Education | M.S. Computer Science | B.S. Computer Science |\n\n"
                "**Recommendation**: Candidate A has a stronger profile for this role based on experience and skill alignment."
            )
        return (
            "## Retrieved Information\n\n"
            "Based on the available recruitment knowledge, here is the information matching your request:\n\n"
            "The retrieved candidates show strong alignment with the specified requirements. "
            "Key skills identified include Python, AWS, and cloud infrastructure experience. "
            "Let me know if you would like me to:\n"
            "- Summarize a specific candidate's profile\n"
            "- Compare multiple candidates\n"
            "- Search for additional criteria"
        )


class _TextChunk:
    """Unified chunk wrapper for all LLM providers."""
    def __init__(self, content: str):
        self.content = content
