# Candidate Screening Agent - Milestone 3

Advanced multi-stage candidate screening system using LangGraph, LangChain, and intelligent agent orchestration.

## Architecture Overview

### Part A: Agent Architecture (40%)

**Graph-Based Workflow:**
```
START → Parse JD → Extract Requirements → Search Resumes → 
Rank Candidates → Generate Report → Human Feedback → END
```

**Files:**
- `agent_state.py` - State definitions (Candidate, JobRequirements, ScreeningState)
- `matching_agent.py` - LangGraph agent with node definitions and workflow orchestration
- Node Types:
  - `parse_jd`: Initial JD parsing and user acknowledgment
  - `extract_requirements`: Structured requirements extraction using LLM
  - `search_candidates`: RAG-based resume search (Milestone 2 integration)
  - `rank_candidates`: Scoring and shortlisting
  - `generate_report`: Multi-candidate report generation
  - `human_feedback`: Feedback loop entry point

**Key Technologies:**
- LangGraph: Graph-based state machine
- LangChain: LLM integration and tool orchestration
- OpenAI: Parse requirements, generate explanations, analyze candidates

### Part B: Interactive Features (30%)

**Conversational Interface (`agent_cli.py`):**
- Natural language query processing
- Iterative refinement of search criteria
- Context-aware follow-up handling

**Supported Queries:**
- "Find me candidates with React and 3+ years experience"
- "Compare the top 3 matches side by side"
- "Generate interview questions for John Doe"
- "Why did John rank higher than Jane?"
- "Refine the search for only Docker experience"

**Refinement Flow:**
1. Extract refined criteria from query
2. Re-evaluate candidates
3. Explain changes in ranking
4. Update shortlist

### Part C: Advanced Capabilities (30%)

**Multi-Round Screening:**
1. **Initial Round** (`search_candidates`): Retrieve top 20 from full dataset
2. **Analysis Round** (`rank_candidates`): Deep dive into top 10 with strengths/gaps
3. **Final Round** (`analyze_candidate_for_hire`): Hire/no-hire recommendations

**Explainability Features:**
- Candidate strengths extraction
- Gap identification
- Improvement suggestions per candidate
- Confidence-based recommendations (hire/no-hire/maybe)

**Screening Tools (`screening_tools.py`):**
- `extract_requirements()` - Parse JD into structured requirements
- `search_resumes_for_jd()` - RAG-based candidate search
- `compare_candidates()` - Head-to-head comparison
- `generate_interview_questions()` - Tailored interview questions
- `analyze_candidate_for_hire()` - Deep candidate analysis

## Setup

```bash
cd assignment_agent_screening
pip install -r requirements.txt
```

Ensure root `.env` contains:
```env
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
```

## Usage

### Run Interactive CLI
```bash
python agent_cli.py
```

### Programmatic Usage
```python
from matching_agent import ScreeningAgent

agent = ScreeningAgent()

# Initial screening
result = agent.run_initial_screening(job_description)

# Follow-up query
result = agent.process_user_query(result, "Compare top 3 candidates")
```

## Output Examples

### Initial Screening Result
```json
{
  "job_description": "...",
  "job_requirements": {
    "must_have_skills": ["Python", "Machine Learning"],
    "nice_to_have_skills": ["FastAPI"],
    "min_years_experience": 5
  },
  "candidates": [...],
  "shortlist": [
    {
      "candidate_name": "John Doe",
      "match_score": 92,
      "matched_skills": ["Python", "ML"],
      "strengths": ["5+ years ML", "AWS experience"],
      "gaps": ["Limited NLP experience"],
      "recommendation": "hire"
    }
  ]
}
```

### Interactive Query Response
```
User: "Compare the top 3 matches"

Agent: [Comparison analysis]
- Winner: John Doe (better AWS skills)
- Recommendation: Interview John first
```

## State Diagram

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌──────────────┐
│  Parse JD    │ (Acknowledge user input)
└────┬─────────┘
     │
     ▼
┌──────────────────────┐
│Extract Requirements  │ (LLM-based parsing)
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ Search Candidates    │ (RAG integration)
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│  Rank Candidates     │ (Score & shortlist)
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ Generate Report      │ (Format output)
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ Human Feedback Loop  │ (Part B: Interactive queries)
└────┬─────────────────┘
     │
     ▼
┌─────────┐
│   END   │
└─────────┘
```

## Test Scenarios

See `test_scenarios.py` for 5+ conversation flows:
1. Basic screening flow
2. Comparison query
3. Interview question generation
4. Ranking explanation
5. Criteria refinement & re-ranking

## Demo Video (5-6 minutes)

Record the following:
1. Setup and environment configuration
2. Run `python agent_cli.py`
3. Paste sample JD and watch initial screening
4. Execute Part B queries (comparisons, explanations)
5. Show Part C full analysis
6. Demonstrate iterative refinement

## Integration with Previous Milestones

- **Milestone 1**: File system tools (`fs_tools.py`) integrated in resume reading
- **Milestone 2**: RAG + Job Matcher (`resume_rag.py`, `job_matcher.py`) for candidate search
- **Milestone 3**: LangGraph agent orchestration on top for intelligent workflow

## LangChain/LangGraph Features Used

- `ChatOpenAI` for LLM interactions
- `JsonOutputParser` for structured output
- `PromptTemplate` for dynamic prompts
- `StateGraph` for graph-based workflow
- `MessagesState` for conversation tracking
- Tool calling via LLM with JSON parsing
