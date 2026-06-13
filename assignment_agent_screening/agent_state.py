from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from langgraph.graph import MessagesState


@dataclass
class Candidate:
    """Candidate profile from resume matching."""

    candidate_name: str
    resume_path: str
    match_score: float
    matched_skills: List[str]
    relevant_excerpts: List[str]
    reasoning: str

    # Added during screening
    experience_years: int = 0
    education: str = ""
    strengths: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    screening_round: int = 1
    interview_questions: List[str] = field(default_factory=list)
    recommendation: str = ""  # hire, no-hire, maybe

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "resume_path": self.resume_path,
            "match_score": self.match_score,
            "matched_skills": self.matched_skills,
            "relevant_excerpts": self.relevant_excerpts,
            "reasoning": self.reasoning,
            "experience_years": self.experience_years,
            "education": self.education,
            "strengths": self.strengths,
            "gaps": self.gaps,
            "improvement_suggestions": self.improvement_suggestions,
            "screening_round": self.screening_round,
            "interview_questions": self.interview_questions,
            "recommendation": self.recommendation,
        }


@dataclass
class JobRequirements:
    """Parsed job requirements."""

    must_have_skills: List[str] = field(default_factory=list)
    nice_to_have_skills: List[str] = field(default_factory=list)
    min_years_experience: int = 0
    preferred_education: str = ""
    other_requirements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "must_have_skills": self.must_have_skills,
            "nice_to_have_skills": self.nice_to_have_skills,
            "min_years_experience": self.min_years_experience,
            "preferred_education": self.preferred_education,
            "other_requirements": self.other_requirements,
        }


class ScreeningState(MessagesState):
    """Agent state for screening workflow."""

    job_description: str = ""
    job_requirements: JobRequirements | None = None
    candidates: List[Candidate] = field(default_factory=list)
    shortlist: List[Candidate] = field(default_factory=list)
    current_round: int = 1
    user_query: str = ""
    agent_response: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
