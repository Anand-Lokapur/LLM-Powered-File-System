"""Proxy module for importing the Milestone 2 resume RAG from the agent folder.

This lets `assignment_rag_job_matching.job_matcher` import `resume_rag` when the
agent is launched from `assignment_agent_screening/`.
"""

from assignment_rag_job_matching.resume_rag import *  # noqa: F401,F403
