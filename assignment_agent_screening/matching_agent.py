from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command

from agent_state import ScreeningState, Candidate, JobRequirements
from screening_tools import (
    extract_requirements,
    search_resumes_for_jd,
    generate_interview_questions,
    compare_candidates,
    analyze_candidate_for_hire,
)


class ScreeningAgent:
    """
    LangGraph-based agent for multi-round candidate screening.
    Part A: Graph-based workflow orchestration.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
        self.graph = self._build_graph()

    def _emit_step(self, trace: list[str], message: str) -> None:
        """Log a workflow step both to trace history and to the console."""
        trace.append(message)
        print(message, flush=True)

    def _shortlist_context(self, shortlist: list[Any], limit: int = 5) -> str:
        """Serialize shortlist into a compact context string for prompts."""
        if not shortlist:
            return "No candidates shortlisted yet."

        lines = []
        for idx, candidate in enumerate(shortlist[:limit], 1):
            name = candidate.candidate_name if hasattr(candidate, "candidate_name") else candidate.get("candidate_name", "N/A")
            score = candidate.match_score if hasattr(candidate, "match_score") else candidate.get("match_score", 0)
            skills = candidate.matched_skills if hasattr(candidate, "matched_skills") else candidate.get("matched_skills", [])
            strengths = candidate.strengths if hasattr(candidate, "strengths") else candidate.get("strengths", [])
            gaps = candidate.gaps if hasattr(candidate, "gaps") else candidate.get("gaps", [])
            lines.append(
                "{idx}. {name} | score={score} | skills={skills} | strengths={strengths} | gaps={gaps}".format(
                    idx=idx,
                    name=name,
                    score=score,
                    skills=", ".join(skills[:5]) or "N/A",
                    strengths=", ".join(strengths[:3]) or "N/A",
                    gaps=", ".join(gaps[:3]) or "N/A",
                )
            )
        return "\n".join(lines)

    def _query_mentions_candidate(self, query: str, shortlist: list[Any]) -> bool:
        """Check whether query mentions any shortlisted candidate names."""
        q = query.lower()
        for candidate in shortlist:
            name = candidate.candidate_name if hasattr(candidate, "candidate_name") else candidate.get("candidate_name", "")
            if name and name.lower() in q:
                return True
        return False

    def _parse_jd_node(self, state: ScreeningState) -> dict[str, Any]:
        """START → Parse JD"""
        user_query = state.get("user_query") or state.get("job_description", "")
        history = list(state.get("conversation_history", []))
        trace = list(state.get("trace", []))
        history.append({"role": "user", "content": user_query})
        self._emit_step(trace, f"[parse_jd] received JD (chars={len(user_query)})")
        self._emit_step(trace, "[parse_jd] analyzing job description with LLM")

        response = self.llm.invoke(
            [
                HumanMessage(
                    content=f"User asking about job: {user_query}. Acknowledge and prepare to parse."
                )
            ]
        )

        history.append({"role": "assistant", "content": response.content})
        self._emit_step(trace, "[parse_jd] acknowledged user request")

        return {
            "job_description": state.get("job_description", "") or user_query,
            "conversation_history": history,
            "trace": trace,
            "agent_response": response.content,
        }

    def _extract_requirements_node(self, state: ScreeningState) -> dict[str, Any]:
        """Parse JD → Extract Requirements"""
        jd = state.get("job_description", "")
        history = list(state.get("conversation_history", []))
        trace = list(state.get("trace", []))
        self._emit_step(trace, "[extract_requirements] parsing JD into must-have / nice-to-have requirements")

        try:
            requirements = extract_requirements(jd)
        except Exception as e:
            requirements = JobRequirements()
            history.append(
                {
                    "role": "assistant",
                    "content": f"Could not parse requirements: {str(e)}",
                }
            )

        response = self.llm.invoke(
            [
                HumanMessage(
                    content=f"Extracted requirements: Must-have: {requirements.must_have_skills[:3]}, Min years: {requirements.min_years_experience}"
                )
            ]
        )

        history.append({"role": "assistant", "content": response.content})
        self._emit_step(trace, f"[extract_requirements] must-have={requirements.must_have_skills[:5]} min_years={requirements.min_years_experience}")

        return {
            "job_requirements": requirements,
            "agent_response": response.content,
            "conversation_history": history,
            "trace": trace,
        }

    def _search_candidates_node(self, state: ScreeningState) -> dict[str, Any]:
        """Extract Requirements → Search Resumes"""
        jd = state.get("job_description", "")
        history = list(state.get("conversation_history", []))
        trace = list(state.get("trace", []))
        self._emit_step(trace, "[search_candidates] querying RAG resume index")

        try:
            candidates = search_resumes_for_jd(jd, top_k=20)
        except Exception as e:
            candidates = []
            history.append(
                {
                    "role": "assistant",
                    "content": f"Error searching resumes: {str(e)}",
                }
            )

        response = self.llm.invoke(
            [
                HumanMessage(
                    content=f"Found {len(candidates)} matching candidates. Top match: {candidates[0].candidate_name if candidates else 'None'} with score {candidates[0].match_score if candidates else 0}."
                )
            ]
        )

        history.append({"role": "assistant", "content": response.content})
        self._emit_step(trace, f"[search_candidates] retrieved={len(candidates)} candidates")
        if not candidates:
            self._emit_step(trace, "[search_candidates] no candidates found; index may have been empty before auto-rebuild")

        return {
            "candidates": candidates,
            "agent_response": response.content,
            "conversation_history": history,
            "trace": trace,
        }

    def _rank_candidates_node(self, state: ScreeningState) -> dict[str, Any]:
        """Search Resumes → Rank Candidates"""
        candidates = sorted(state.get("candidates", []), key=lambda c: c.match_score, reverse=True)[:10]
        job_requirements = state.get("job_requirements") or JobRequirements()
        history = list(state.get("conversation_history", []))
        trace = list(state.get("trace", []))
        self._emit_step(trace, f"[rank_candidates] ranking top {len(state.get('candidates', []))} candidates")

        shortlist = []
        for idx, candidate in enumerate(candidates, 1):
            candidate.screening_round = 1
            shortlist.append(candidate)

            if idx < 4:
                try:
                    analysis = analyze_candidate_for_hire(candidate, job_requirements)
                    candidate.strengths = analysis.get("strengths", [])
                    candidate.gaps = analysis.get("gaps", [])
                    candidate.improvement_suggestions = analysis.get(
                        "improvement_suggestions", []
                    )
                except Exception:
                    pass

        response = self.llm.invoke(
            [
                HumanMessage(
                    content=f"Ranked top {len(shortlist)} candidates. Ready to generate detailed report."
                )
            ]
        )

        history.append({"role": "assistant", "content": response.content})
        self._emit_step(trace, f"[rank_candidates] shortlist={len(shortlist)}")

        return {
            "shortlist": shortlist,
            "current_round": 2,
            "agent_response": response.content,
            "conversation_history": history,
            "trace": trace,
        }

    def _generate_report_node(self, state: ScreeningState) -> dict[str, Any]:
        """Rank Candidates → Generate Report"""
        shortlist = state.get("shortlist", [])[:5]
        candidates = state.get("candidates", [])
        history = list(state.get("conversation_history", []))
        trace = list(state.get("trace", []))
        self._emit_step(trace, f"[generate_report] generating report for {len(state.get('shortlist', []))} shortlisted candidates")

        report_lines = ["# Screening Report\n"]
        report_lines.append(f"Total Candidates: {len(candidates)}\n")
        report_lines.append(f"Shortlist Size: {len(shortlist)}\n\n")

        for idx, candidate in enumerate(shortlist, 1):
            report_lines.append(f"## {idx}. {candidate.candidate_name}\n")
            report_lines.append(f"- Score: {candidate.match_score}/100\n")
            report_lines.append(f"- Skills: {', '.join(candidate.matched_skills[:5])}\n")
            if candidate.strengths:
                report_lines.append(f"- Strengths: {', '.join(candidate.strengths)}\n")
            if candidate.gaps:
                report_lines.append(f"- Gaps: {', '.join(candidate.gaps)}\n")
            report_lines.append("\n")

        report = "".join(report_lines)

        response = self.llm.invoke(
            [HumanMessage(content=f"Generated report with {len(shortlist)} candidates")]
        )

        history.append({"role": "assistant", "content": response.content})
        self._emit_step(trace, "[generate_report] report ready")

        return {
            "agent_response": report + "\n\n" + response.content,
            "conversation_history": history,
            "trace": trace,
        }


    def _human_feedback_node(self, state: ScreeningState) -> dict[str, Any]:
        """Generate Report ? Human Feedback Loop"""
        trace = list(state.get("trace", []))
        shortlist = state.get("shortlist", [])
        self._emit_step(trace, "[human_feedback] waiting for user follow-up")

        if not shortlist:
            response_text = (
                "No candidates were shortlisted. The agent completed screening, but the RAG search returned no matches. "
                "Please ensure the resume index is built and the resumes folder contains supported files."
            )
        else:
            response_text = (
                "Ready for follow-up queries. You can: (1) Ask to compare candidates, (2) Request interview questions, "
                "(3) Refine criteria, or (4) Proceed to final round."
            )

        return {
            "agent_response": response_text,
            "conversation_history": list(state.get("conversation_history", [])),
            "trace": trace,
        }

    def _build_graph(self):
        """Build LangGraph workflow: Part A - Agent Architecture"""
        graph = StateGraph(ScreeningState)

        graph.add_node("parse_jd", self._parse_jd_node)
        graph.add_node("extract_requirements", self._extract_requirements_node)
        graph.add_node("search_candidates", self._search_candidates_node)
        graph.add_node("rank_candidates", self._rank_candidates_node)
        graph.add_node("generate_report", self._generate_report_node)
        graph.add_node("human_feedback", self._human_feedback_node)

        graph.add_edge(START, "parse_jd")
        graph.add_edge("parse_jd", "extract_requirements")
        graph.add_edge("extract_requirements", "search_candidates")
        graph.add_edge("search_candidates", "rank_candidates")
        graph.add_edge("rank_candidates", "generate_report")
        graph.add_edge("generate_report", "human_feedback")
        graph.add_edge("human_feedback", END)

        return graph.compile()

    def run_initial_screening(self, job_description: str) -> dict:
        """Execute initial screening workflow for a job description."""
        print("[screening] starting initial screening pipeline...", flush=True)
        state = {
            "messages": [],
            "job_description": job_description,
            "user_query": job_description,
            "conversation_history": [],
            "trace": [],
            "candidates": [],
            "shortlist": [],
            "current_round": 1,
            "agent_response": "",
        }

        result = self.graph.invoke(state)

        print("[screening] initial screening pipeline finished.", flush=True)

        return {
            "job_description": result.get("job_description", ""),
            "job_requirements": result.get("job_requirements"),
            "candidates": result.get("candidates", []),
            "shortlist": result.get("shortlist", []),
            "agent_response": result.get("agent_response", ""),
            "conversation_history": result.get("conversation_history", []),
            "trace": result.get("trace", []),
        }

    def process_user_query(self, state_dict: dict, query: str) -> dict:
        """
        Part B: Handle interactive queries and iterative refinement.
        Process follow-up questions like "Find candidates with React and 3+ years"
        """
        state = {
            "messages": [],
            "job_description": state_dict.get("job_description", ""),
            "job_requirements": state_dict.get("job_requirements"),
            "candidates": state_dict.get("candidates", []),
            "shortlist": state_dict.get("shortlist", []),
            "user_query": query,
            "conversation_history": state_dict.get("conversation_history", []),
        }

        q = query.lower()
        shortlist = state.get("shortlist", [])

        if "compare" in q:
            return self._handle_compare_query(state, query)
        elif "question" in q or "interview" in q:
            return self._handle_interview_query(state, query)
        elif (
            "why" in q
            or "how" in q
            or "less" in q
            or "higher" in q
            or "lower" in q
            or "rank" in q
            or self._query_mentions_candidate(query, shortlist)
        ):
            return self._handle_explanation_query(state, query)
        elif "refine" in q or "adjust" in q:
            return self._handle_refinement_query(state, query)
        else:
            return self._handle_generic_query(state, query)

    def _handle_compare_query(self, state: dict[str, Any], query: str) -> dict:
        """Part B: Handle comparison queries"""
        candidates_to_compare = [c.candidate_name for c in state.get("shortlist", [])[:3]]

        try:
            comparison = compare_candidates(candidates_to_compare, state.get("shortlist", []))
            response = json.dumps(comparison, indent=2)
        except Exception as e:
            response = f"Error comparing candidates: {str(e)}"

        history = list(state.get("conversation_history", []))
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})

        return {
            "agent_response": response,
            "conversation_history": history,
        }

    def _handle_interview_query(self, state: dict[str, Any], query: str) -> dict:
        """Part C: Generate interview questions for candidate"""
        shortlist = state.get("shortlist", [])
        candidate_name = shortlist[0].candidate_name if shortlist else None

        if not candidate_name:
            response = "No candidates in shortlist to generate questions for."
        else:
            try:
                questions = generate_interview_questions(
                    candidate_name,
                    shortlist,
                    state.get("job_requirements") or JobRequirements(),
                )
                response = json.dumps(questions, indent=2)
            except Exception as e:
                response = f"Error generating questions: {str(e)}"

        history = list(state.get("conversation_history", []))
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})

        return {
            "agent_response": response,
            "conversation_history": history,
        }

    def _handle_explanation_query(self, state: dict[str, Any], query: str) -> dict:
        """Part B: Explain ranking differences"""
        shortlist = state.get("shortlist", [])
        shortlist_context = self._shortlist_context(shortlist, limit=5)
        prompt = f"""User asks: {query}

Current shortlist context:
{shortlist_context}

Explain why a candidate ranked higher using the shortlist context. If a specific candidate is mentioned, compare that candidate to the others in the shortlist.
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])

        history = list(state.get("conversation_history", []))
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response.content})

        return {
            "agent_response": response.content,
            "conversation_history": history,
        }

    def _handle_refinement_query(self, state: dict[str, Any], query: str) -> dict:
        """Part B: Handle refinement of criteria and re-rank"""
        response = self.llm.invoke(
            [
                HumanMessage(
                    content=f"User refining criteria: {query}\n\nCurrent shortlist: {[c.candidate_name for c in state.get('shortlist', [])]}\n\nProvide re-ranking based on new criteria."
                )
            ]
        )

        history = list(state.get("conversation_history", []))
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response.content + "\n(Re-ranking applied)"})

        return {
            "agent_response": response.content,
            "conversation_history": history,
        }

    def _handle_generic_query(self, state: dict[str, Any], query: str) -> dict:
        """Handle generic conversational queries"""
        shortlist = state.get("shortlist", [])
        shortlist_context = self._shortlist_context(shortlist, limit=5)
        job_description = state.get("job_description", "")
        prompt = f"""User query: {query}\n\nJob description:\n{job_description}\n\nCurrent shortlist context:\n{shortlist_context}\n\nUse the shortlist context to answer the user's question. If they ask about a candidate ranking, compare the candidates in the shortlist.\n"""
        response = self.llm.invoke([HumanMessage(content=prompt)])

        history = list(state.get("conversation_history", []))
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response.content})

        return {
            "agent_response": response.content,
            "conversation_history": history,
        }
    def get_graph(self):
        """Return the graph for visualization/introspection."""
        return self.graph
