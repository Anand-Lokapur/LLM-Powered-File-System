from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

try:
    from .agent_state import Candidate, JobRequirements
except ImportError:
    from agent_state import Candidate, JobRequirements


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from assignment_rag_job_matching.job_matcher import JobMatcher
from assignment_rag_job_matching.resume_rag import ResumeRAG


load_dotenv()


def extract_requirements(job_description: str) -> JobRequirements:
    """
    Use LLM to parse job description into structured requirements.
    Part A: Extract must-have vs nice-to-have skills and other requirements.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    parser = JsonOutputParser(pydantic_object=dict)

    prompt = PromptTemplate(
        template="""
Analyze this job description and extract requirements:

Job Description:
{job_description}

Extract and return as JSON:
{{
    "must_have_skills": ["list of critical skills required"],
    "nice_to_have_skills": ["list of preferred but not required skills"],
    "min_years_experience": 0,
    "preferred_education": "education requirement if any",
    "other_requirements": ["other key requirements like certifications, languages, etc"]
}}

Return ONLY valid JSON.
""",
        input_variables=["job_description"],
        partial_variables={},
    )

    chain = prompt | llm | parser

    result = chain.invoke({"job_description": job_description})

    return JobRequirements(
        must_have_skills=result.get("must_have_skills", []),
        nice_to_have_skills=result.get("nice_to_have_skills", []),
        min_years_experience=int(result.get("min_years_experience", 0)),
        preferred_education=result.get("preferred_education", ""),
        other_requirements=result.get("other_requirements", []),
    )


def compare_candidates(candidate_ids: List[str], candidates: List[Candidate]) -> Dict[str, Any]:
    """
    Part B: Head-to-head comparison of multiple candidates.
    Returns structured comparison with strengths/gaps for each.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    candidates_info = [c for c in candidates if c.candidate_name in candidate_ids]
    if not candidates_info:
        return {"error": "No matching candidates found"}

    candidate_json = json.dumps([c.to_dict() for c in candidates_info], indent=2)

    parser = JsonOutputParser(pydantic_object=dict)

    prompt = PromptTemplate(
        template="""
Compare these candidates side-by-side:

{candidates_json}

Provide comparison in JSON format:
{{
    "candidates_compared": ["names"],
    "winner": "name of best match",
    "comparison_points": {{
        "skills_alignment": "analysis",
        "experience": "analysis",
        "gaps": "analysis"
    }},
    "recommendation": "which candidate to interview first and why"
}}

Return ONLY valid JSON.
""",
        input_variables=["candidates_json"],
    )

    chain = prompt | llm | parser

    result = chain.invoke({"candidates_json": candidate_json})

    return result


def generate_interview_questions(candidate_id: str, candidates: List[Candidate], job_requirements: JobRequirements | None = None) -> Dict[str, Any]:
    """
    Part C: Generate screening questions tailored to candidate's resume.
    Uses LLM to create role-specific and gap-filling questions.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    candidate = next((c for c in candidates if c.candidate_name == candidate_id), None)
    if not candidate:
        return {"error": f"Candidate {candidate_id} not found"}

    candidate_json = json.dumps(candidate.to_dict(), indent=2)
    requirements_json = json.dumps((job_requirements or JobRequirements()).to_dict(), indent=2)

    parser = JsonOutputParser(pydantic_object=dict)

    prompt = PromptTemplate(
        template="""
Generate screening interview questions for this candidate:

Candidate Profile:
{candidate_json}

Job Requirements:
{requirements_json}

Create JSON with interview questions:
{{
    "candidate_name": "name",
    "technical_questions": ["5-7 technical questions based on skills"],
    "behavioral_questions": ["5 behavioral questions"],
    "gap_filling_questions": ["2-3 questions about missing skills or experience"],
    "strengths_to_explore": ["questions to dive deeper into strengths"],
    "estimated_level": "junior/mid/senior"
}}

Return ONLY valid JSON.
""",
        input_variables=["candidate_json", "requirements_json"],
    )

    chain = prompt | llm | parser

    result = chain.invoke(
        {"candidate_json": candidate_json, "requirements_json": requirements_json}
    )

    if "interview_questions" not in result:
        result["interview_questions"] = (
            result.get("technical_questions", [])
            + result.get("behavioral_questions", [])
            + result.get("gap_filling_questions", [])
        )

    return result


def search_resumes_for_jd(job_description: str, top_k: int = 10) -> List[Candidate]:
    """
    Use Milestone 2 RAG + Job Matcher to search resumes for JD.
    Returns Candidate objects.
    """
    rag = ResumeRAG(
        resumes_dir=str((ROOT_DIR / "sample_data" / "resumes").resolve()),
        persist_dir=str((CURRENT_DIR / "vector_store").resolve()),
    )

    matcher = JobMatcher(rag)
    output = matcher.match_job(job_description=job_description, top_k=top_k)
    top_matches = output.get("top_matches", [])

    if not top_matches:
        try:
            rag.rebuild_index()
            output = matcher.match_job(job_description=job_description, top_k=top_k)
            top_matches = output.get("top_matches", [])
        except Exception:
            top_matches = []

    candidates = []
    for match in top_matches:
        candidates.append(
            Candidate(
                candidate_name=match.get("candidate_name", ""),
                resume_path=match.get("resume_path", ""),
                match_score=float(match.get("match_score", 0)),
                matched_skills=match.get("matched_skills", []),
                relevant_excerpts=match.get("relevant_excerpts", []),
                reasoning=match.get("reasoning", ""),
            )
        )

    return candidates


def analyze_candidate_for_hire(candidate: Candidate, job_requirements: JobRequirements) -> Dict[str, Any]:
    """
    Part C: Deep analysis of candidate for hire/no-hire recommendation.
    Generates strengths, gaps, and improvement suggestions.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    candidate_json = json.dumps(candidate.to_dict(), indent=2)
    requirements_json = json.dumps(job_requirements.to_dict(), indent=2)

    parser = JsonOutputParser(pydantic_object=dict)

    prompt = PromptTemplate(
        template="""
Analyze this candidate for the role:

Candidate:
{candidate_json}

Requirements:
{requirements_json}

Provide analysis in JSON:
{{
    "candidate_name": "name",
    "strengths": ["2-3 key strengths"],
    "gaps": ["2-3 key gaps"],
    "improvement_suggestions": ["2-3 suggestions to address gaps"],
    "recommendation": "hire/no-hire/maybe",
    "confidence": 0.0-1.0,
    "reasoning": "brief reasoning for recommendation"
}}

Return ONLY valid JSON.
""",
        input_variables=["candidate_json", "requirements_json"],
    )

    chain = prompt | llm | parser

    result = chain.invoke(
        {"candidate_json": candidate_json, "requirements_json": requirements_json}
    )

    return result
