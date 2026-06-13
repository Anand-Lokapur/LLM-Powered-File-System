from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .job_matcher import JobMatcher
    from .resume_rag import ResumeRAG, default_persist_dir, default_resumes_dir
except ImportError:
    from job_matcher import JobMatcher
    from resume_rag import ResumeRAG, default_persist_dir, default_resumes_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click demo runner for resume RAG + matching")
    parser.add_argument("--resumes-dir", default=default_resumes_dir(), help="Directory containing resumes")
    parser.add_argument("--persist-dir", default=default_persist_dir(), help="Vector DB persistence directory")
    parser.add_argument(
        "--job-file",
        default=str((Path(__file__).resolve().parent / "sample_job_description.txt").resolve()),
        help="Path to job description file",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top matches to return")
    args = parser.parse_args()

    print("[1/3] Building RAG index...")
    rag = ResumeRAG(resumes_dir=args.resumes_dir, persist_dir=args.persist_dir)
    index_result = rag.rebuild_index()
    print(json.dumps(index_result, indent=2))

    if not index_result.get("success"):
        return

    print("\n[2/3] Loading job description...")
    job_description = Path(args.job_file).read_text(encoding="utf-8").strip()
    print(f"Job description chars: {len(job_description)}")

    print("\n[3/3] Running job matching...")
    matcher = JobMatcher(rag)
    output = matcher.match_job(job_description=job_description, top_k=args.top_k)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
