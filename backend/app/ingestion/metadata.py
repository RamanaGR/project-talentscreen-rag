"""Metadata Extraction Service (PDF Part 4.5).

Extracts structured metadata from documents using LLM with regex-based fallback.
Supports multiple document types: resumes, job descriptions, interview guidelines,
hiring policies, recruitment SOPs, and internal knowledge documents.
"""

import json
import re
from typing import Dict, Any, List
from app.core.logging import logger
from app.services.llm_service import LLMService


class MetadataExtractor:
    """Extracts metadata from documents using LLM with regex fallback.
    
    Metadata includes: skills, years_of_experience, education, domain,
    candidate_name, job_role, certifications, document_type.
    """

    def __init__(self):
        self.llm_service = LLMService()

    def detect_document_type(self, text: str) -> str:
        """Detect document type from content analysis."""
        text_lower = text.lower()
        
        # Job Description indicators
        jd_indicators = ["job description", "responsibilities", "requirements", "qualifications",
                         "we are looking for", "role overview", "about the role", "what you'll do"]
        jd_score = sum(1 for ind in jd_indicators if ind in text_lower)
        
        # Interview Guidelines indicators
        interview_indicators = ["interview", "evaluation criteria", "assessment", "interview questions",
                               "interview process", "scoring rubric"]
        interview_score = sum(1 for ind in interview_indicators if ind in text_lower)
        
        # Hiring Policy indicators
        policy_indicators = ["policy", "compliance", "hiring standards", "approval process",
                           "equal opportunity", "background check"]
        policy_score = sum(1 for ind in policy_indicators if ind in text_lower)
        
        # SOP indicators
        sop_indicators = ["standard operating procedure", "sop", "workflow", "escalation",
                         "process flow", "step-by-step"]
        sop_score = sum(1 for ind in sop_indicators if ind in text_lower)
        
        # Resume indicators
        resume_indicators = ["experience", "education", "skills", "summary", "certifications",
                           "projects", "work history", "professional experience"]
        resume_score = sum(1 for ind in resume_indicators if ind in text_lower)
        
        scores = {
            "job_description": jd_score,
            "interview_guide": interview_score,
            "hiring_policy": policy_score,
            "sop": sop_score,
            "resume": resume_score
        }
        
        best_type = max(scores, key=scores.get)
        if scores[best_type] == 0:
            return "knowledge_doc"
        return best_type

    def extract_metadata(self, text: str, document_type: str = None) -> Dict[str, Any]:
        """Extract metadata using LLM if available, otherwise regex fallback."""
        if not document_type:
            document_type = self.detect_document_type(text)
        
        # Try the configured provider before using the deterministic fallback.
        try:
            return self._extract_with_llm(text, document_type)
        except Exception as e:
            logger.warning(f"LLM metadata extraction failed: {e}. Falling back to regex.")
        
        # Regex-based fallback
        return self._extract_with_regex(text, document_type)

    def _extract_with_llm(self, text: str, document_type: str) -> Dict[str, Any]:
        """Extract metadata using the configured LLM provider."""
        prompt = f"""Extract professional metadata from the following {document_type} text.
Return ONLY a valid JSON object with these fields:
- "document_type": "{document_type}"
- "skills": list of technical skills found
- "years_of_experience": total years as integer (0 if not applicable)
- "education": highest degree and institution (empty string if not found)
- "domain": primary industry/field (e.g., Fintech, Healthcare, AI, Engineering)
- "candidate_name": full name of the person (empty string if not a resume)
- "job_role": primary job title or role mentioned
- "certifications": list of certifications found
- "projects": list of notable project names

Document Text (first 3000 chars):
{text[:3000]}
"""
        text_response = self.llm_service.invoke("", prompt)
        
        # Extract JSON from response
        match = re.search(r'\{.*\}', text_response, re.DOTALL)
        if match:
            result = json.loads(match.group())
            result["document_type"] = document_type
            return result
        
        return self._extract_with_regex(text, document_type)

    def _extract_with_regex(self, text: str, document_type: str) -> Dict[str, Any]:
        """Regex-based metadata extraction fallback."""
        metadata = {
            "document_type": document_type,
            "skills": self._extract_skills(text),
            "years_of_experience": self._extract_experience_years(text),
            "education": self._extract_education(text),
            "domain": "",
            "candidate_name": self._extract_name(text),
            "job_role": self._extract_job_role(text),
            "certifications": self._extract_certifications(text),
            "projects": []
        }
        return metadata

    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical skills using keyword matching."""
        known_skills = [
            "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++", "C#",
            "React", "Angular", "Vue", "Node.js", "Django", "Flask", "FastAPI",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
            "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
            "Snowflake", "Databricks", "Spark", "Airflow", "Kafka", "dbt",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
            "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
            "Git", "CI/CD", "Jenkins", "GitHub Actions", "Linux",
            "REST API", "GraphQL", "Microservices", "ETL",
            "Tableau", "Power BI", "Excel", "Agile", "Scrum"
        ]
        found_skills = []
        text_lower = text.lower()
        for skill in known_skills:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        return found_skills

    def _extract_experience_years(self, text: str) -> int:
        """Extract years of experience from text."""
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
            r'experience\s*[:]\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s+in\s+',
        ]
        max_years = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                years = int(match)
                if 0 < years < 50:
                    max_years = max(max_years, years)
        return max_years

    def _extract_education(self, text: str) -> str:
        """Extract education level."""
        degrees = [
            (r"Ph\.?D\.?|Doctor(?:ate)?", "PhD"),
            (r"Master(?:'s)?|M\.?S\.?|M\.?Sc\.?|MBA|M\.?Tech", "Master's"),
            (r"Bachelor(?:'s)?|B\.?S\.?|B\.?Sc\.?|B\.?Tech|B\.?E\.?", "Bachelor's"),
        ]
        for pattern, degree in degrees:
            if re.search(pattern, text, re.IGNORECASE):
                return degree
        return ""

    def _extract_name(self, text: str) -> str:
        """Try to extract candidate name from the beginning of a resume."""
        lines = text.strip().split('\n')
        for line in lines[:5]:
            line = line.strip()
            # Skip empty lines and common headers
            if not line or len(line) < 3 or len(line) > 60:
                continue
            # Check if it looks like a name (2-4 capitalized words)
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
                # Skip common headers
                skip_words = {"resume", "curriculum", "vitae", "summary", "profile", "experience", "education"}
                if not any(w.lower() in skip_words for w in words):
                    return line
        return ""

    def _extract_job_role(self, text: str) -> str:
        """Extract primary job role from text."""
        role_patterns = [
            r'(?:Senior|Lead|Principal|Staff|Junior|Mid)?\s*(?:Software|Data|ML|AI|Cloud|DevOps|Full[\s-]?Stack|Front[\s-]?End|Back[\s-]?End)\s*Engineer(?:ing)?',
            r'(?:Senior|Lead|Principal)?\s*(?:Data|Business|Product)\s*(?:Scientist|Analyst|Manager)',
            r'(?:Senior|Lead|Principal)?\s*(?:Machine Learning|Deep Learning|NLP)\s*(?:Engineer|Researcher)',
            r'(?:Senior|Lead)?\s*(?:Python|Java|React|Node\.js)\s*Developer',
        ]
        for pattern in role_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group().strip()
        return ""

    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications."""
        cert_patterns = [
            r'AWS\s+Certified\s+[\w\s-]+',
            r'Google\s+Cloud\s+(?:Professional|Associate)\s+[\w\s-]+',
            r'Azure\s+(?:Certified|Administrator|Developer)\s*[\w\s-]*',
            r'PMP|ITIL|CISSP|CEH|CompTIA\s+\w+',
            r'Certified\s+(?:Kubernetes|Scrum|Data)\s+\w+',
            r'SnowPro\s+\w+',
        ]
        certs = []
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certs.extend([m.strip() for m in matches])
        return certs

    @staticmethod
    def normalize_skills(skills: List[str]) -> List[str]:
        return [s.lower().strip() for s in skills]
