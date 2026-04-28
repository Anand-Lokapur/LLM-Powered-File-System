from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from resume_rag import ResumeRAG, default_persist_dir, default_resumes_dir


CRITICAL_SKILLS = {
    "python",
    "java",
    "javascript",
    "sql",
    "machine learning",
    "nlp",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "react",
    "node.js",
    "fastapi",
    "django",
    "pytorch",
    "scikit-learn",
    "selenium",
    "playwright",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_required_years(job_description: str) -> int:
    patterns = [
        r"(\d{1,2})\+\s*years",
        r"minimum\s+(\d{1,2})\s*years",
        r"at\s*least\s*(\d{1,2})\s*years",
    ]
    values: List[int] = []
    for pattern in patterns:
        matches = re.findall(pattern, job_description, flags=re.IGNORECASE)
        values.extend(int(match) for match in matches)
    return max(values) if values else 0


def _extract_jd_skills(job_description: str) -> List[str]:
    text = _normalize(job_description)
    return sorted([skill for skill in CRITICAL_SKILLS if skill in text])


class JobMatcher:
    def __init__(self, rag: ResumeRAG) -> None:
        self.rag = rag

    def _semantic_score(self, distance: float) -> float:
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        return similarity

    def _keyword_score(self, jd_skills: List[str], candidate_skills: List[str]) -> float:
        if not jd_skills:
            return 0.5
        if not candidate_skills:
            return 0.0
        overlap = len(set(jd_skills).intersection(set(candidate_skills)))
        return overlap / max(len(jd_skills), 1)

    def _candidate_from_chunk_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        skills_csv = metadata.get("skills_csv", "")
        skills = [value.strip() for value in skills_csv.split(",") if value.strip()]
        return {
            "candidate_name": metadata.get("candidate_name", "Unknown Candidate"),
            "resume_path": metadata.get("resume_path", ""),
            "skills": skills,
            "experience_years": int(metadata.get("experience_years", 0)),
            "education": metadata.get("education", ""),
        }

    def match_job(self, job_description: str, top_k: int = 10) -> Dict[str, Any]:
        jd_skills = _extract_jd_skills(job_description)
        min_years = _extract_required_years(job_description)

        retrieval = self.rag.query_chunks(job_description, n_results=max(top_k * 5, 20))
        docs = retrieval.get("documents", [[]])[0]
        metadatas = retrieval.get("metadatas", [[]])[0]
        distances = retrieval.get("distances", [[]])[0]

        grouped: Dict[str, Dict[str, Any]] = {}
        for doc, metadata, distance in zip(docs, metadatas, distances):
            candidate = self._candidate_from_chunk_metadata(metadata)
            resume_path = candidate["resume_path"]
            if not resume_path:
                continue

            semantic = self._semantic_score(distance)
            entry = grouped.setdefault(
                resume_path,
                {
                    "candidate_name": candidate["candidate_name"],
                    "resume_path": resume_path,
                    "skills": candidate["skills"],
                    "experience_years": candidate["experience_years"],
                    "semantic_scores": [],
                    "excerpts": [],
                },
            )
            entry["semantic_scores"].append(semantic)
            if len(entry["excerpts"]) < 3 and doc:
                entry["excerpts"].append(doc[:260])

        candidates: List[Dict[str, Any]] = []
        for _, entry in grouped.items():
            matched_skills = sorted(set(jd_skills).intersection(set(entry["skills"])))

            if min_years > 0 and entry["experience_years"] < min_years:
                continue

            semantic_component = max(entry["semantic_scores"]) if entry["semantic_scores"] else 0.0
            keyword_component = self._keyword_score(jd_skills, entry["skills"])
            score = int(round((semantic_component * 0.7 + keyword_component * 0.3) * 100))

            if min_years > 0:
                score += 5
            score = max(0, min(100, score))

            reasoning_parts = []
            if matched_skills:
                reasoning_parts.append(f"Matched critical skills: {', '.join(matched_skills)}")
            if entry["experience_years"]:
                reasoning_parts.append(f"Estimated experience: {entry['experience_years']} years")
            reasoning_parts.append("Semantic resume sections align with JD context")

            candidates.append(
                {
                    "candidate_name": entry["candidate_name"],
                    "resume_path": entry["resume_path"],
                    "match_score": score,
                    "matched_skills": matched_skills,
                    "relevant_excerpts": entry["excerpts"],
                    "reasoning": "; ".join(reasoning_parts),
                }
            )

        candidates.sort(key=lambda item: item["match_score"], reverse=True)
        return {
            "job_description": job_description,
            "top_matches": candidates[:top_k],
        }


def _read_job_description(args: argparse.Namespace) -> str:
    if args.job_description:
        return args.job_description.strip()
    if args.job_file:
        return Path(args.job_file).read_text(encoding="utf-8").strip()
    raise ValueError("Provide --job-description or --job-file")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-job matching engine")
    parser.add_argument("--job-description", help="Raw job description text")
    parser.add_argument("--job-file", help="Path to job description text file")
    parser.add_argument("--top-k", type=int, default=10, help="Top K resumes to return")
    parser.add_argument("--resumes-dir", default=default_resumes_dir(), help="Resumes directory")
    parser.add_argument("--persist-dir", default=default_persist_dir(), help="Vector store directory")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuild of vector index before matching",
    )
    args = parser.parse_args()

    job_description = _read_job_description(args)
    rag = ResumeRAG(resumes_dir=args.resumes_dir, persist_dir=args.persist_dir)

    if args.reindex:
        rag.rebuild_index()

    matcher = JobMatcher(rag)
    output = matcher.match_job(job_description=job_description, top_k=args.top_k)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
