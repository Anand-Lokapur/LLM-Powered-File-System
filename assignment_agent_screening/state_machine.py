"""State machine diagram and architecture documentation."""

STATE_MACHINE_ASCII = """
┌─────────────────────────────────────────────────────────────────┐
│               SCREENING AGENT STATE MACHINE                     │
│                    Part A: Architecture                         │
└─────────────────────────────────────────────────────────────────┘

                          ╔═════════════╗
                          ║    START    ║
                          ╚═══════╤═════╝
                                  │
                        ┌─────────┘
                        │  Job Description Input
                        ├─────────┐
                        ▼         │
        ┌───────────────────────┐ │
        │  Parse JD             │ │ - Extract raw text
        │  (parse_jd_node)      │ │ - Acknowledge user
        └───────────┬───────────┘ │ - Prepare workflow
                    │             │
                    └─────────────┘
                        │
              ┌─────────┘
              │
              ▼
┌──────────────────────────────────┐
│Extract Requirements              │
│(extract_requirements_node)       │
│                                  │
│- LLM parsing of JD               │
│- Identify must-haves             │
│- Identify nice-to-haves          │
│- Extract min experience          │
└──────────────┬───────────────────┘
               │
       ┌───────┘
       │
       ▼
┌──────────────────────────────────┐
│ Search Resumes                   │
│ (search_candidates_node)         │
│                                  │
│- RAG query on resume chunks      │
│- Semantic matching               │
│- Get top 20 candidates           │
│- Return with scores              │
└──────────────┬───────────────────┘
               │
       ┌───────┘
       │
       ▼
┌──────────────────────────────────┐
│ Rank Candidates                  │
│ (rank_candidates_node)           │
│                                  │
│- Score candidates                │
│- Shortlist top 10                │
│- Run deep analysis on top 3      │
│- Extract strengths/gaps          │
└──────────────┬───────────────────┘
               │
       ┌───────┘
       │
       ▼
┌──────────────────────────────────┐
│ Generate Report                  │
│ (generate_report_node)           │
│                                  │
│- Format shortlist                │
│- Summarize findings              │
│- Prepare output JSON             │
└──────────────┬───────────────────┘
               │
       ┌───────┘
       │
       ▼
┌──────────────────────────────────┐
│ Human Feedback Loop              │
│ (human_feedback_node)            │
│                                  │  Part B: Interactive Queries
│ INTERACTIVE MODE (Part B)        │  ├─ Compare candidates
│                                  │  ├─ Explain rankings
│ Available Actions:               │  ├─ Generate questions
│ 1. Compare candidates            │  ├─ Refine criteria
│ 2. Ask for explanations          │  ├─ Adjust requirements
│ 3. Generate interview Q's        │  └─ Deep analysis
│ 4. Refine criteria
│ 5. Run deep analysis (Part C)
│ 6. Generate hire/no-hire recs
└──────────────┬───────────────────┘
               │
               │ If "end" or "exit"
               │
       ┌───────┘
       │
       ▼
    ╔═════════╗
    ║   END   ║
    ╚═════════╝


DETAILED STATE STRUCTURE:
═══════════════════════════════════════════════════════════════════

class ScreeningState(MessagesState):
    ├─ job_description: str
    │  └─ Original job posting text
    │
    ├─ job_requirements: JobRequirements | None
    │  ├─ must_have_skills: List[str]
    │  ├─ nice_to_have_skills: List[str]
    │  ├─ min_years_experience: int
    │  ├─ preferred_education: str
    │  └─ other_requirements: List[str]
    │
    ├─ candidates: List[Candidate]
    │  ├─ All retrieved candidates from RAG search
    │  └─ Each contains match_score, skills, reasoning
    │
    ├─ shortlist: List[Candidate]
    │  ├─ Top 10 ranked candidates after filtering
    │  ├─ Includes strengths, gaps, improvements
    │  └─ May have interview_questions attached
    │
    ├─ current_round: int
    │  ├─ 1: Initial screening
    │  ├─ 2: Deep analysis
    │  └─ 3: Final recommendations
    │
    ├─ conversation_history: List[Dict]
    │  └─ [{"role": "user"/"assistant", "content": "..."}, ...]
    │
    ├─ user_query: str (current query)
    └─ agent_response: str (last agent response)


DATA FLOW DIAGRAM (Part A → B → C):
═══════════════════════════════════════════════════════════════════

                    Part A: Architecture
                           │
              ┌────────────┼────────────┐
              │            │            │
         Parse JD     Extract Req'   Search RAG
              │            │            │
              └────────────┼────────────┘
                           │
                        Rank Candidates
                           │
                    ┌───────┴───────┐
                    │               │
                    ▼               ▼
            Part B: Interactive   Part C: Deep
            Query Processing      Analysis
                    │               │
    - Compare       ├─ Refine criteria
    - Explain       ├─ Analyze gaps
    - Questions     ├─ Generate Q's
    - Clarify       ├─ Recommendations
                    │ (hire/no-hire/maybe)
                    │
                    └─ Human loops back
                      to refinement


TOOL INTEGRATION:
═══════════════════════════════════════════════════════════════════

Milestone 1 (File System):
  └─ read_file, list_files, write_file, search_in_file
     (Used indirectly via RAG)

Milestone 2 (RAG):
  └─ ResumeRAG (resume_rag.py)
     ├─ Chunk processing
     ├─ Embedding generation
     ├─ Vector store (ChromaDB)
     └─ Query execution
  
  └─ JobMatcher (job_matcher.py)
     ├─ Semantic search
     ├─ Keyword filtering
     ├─ Scoring
     └─ Output formatting

Milestone 3 (Agent):
  └─ ScreeningAgent (matching_agent.py)
     ├─ extract_requirements (LLM-based)
     ├─ compare_candidates (LLM-based)
     ├─ generate_interview_questions (LLM-based)
     └─ analyze_candidate_for_hire (LLM-based)


PART A → B → C FEATURE MAPPING:
═══════════════════════════════════════════════════════════════════

Part A (40%): Agent Architecture
├─ Graph-based workflow orchestration
├─ State machine with 6 nodes
├─ Tool integration
└─ Multi-stage pipeline

Part B (30%): Interactive Features
├─ Natural language query processing
├─ Iterative refinement with re-ranking
├─ Context-aware follow-ups
├─ Explanation generation
└─ Comparison capabilities

Part C (30%): Advanced Capabilities
├─ Multi-round screening (initial → analysis → final)
├─ Deep candidate analysis
├─ Hire/no-hire recommendations
├─ Gap and strength identification
├─ Improvement suggestions
└─ Explainability reports
"""

if __name__ == "__main__":
    print(STATE_MACHINE_ASCII)
