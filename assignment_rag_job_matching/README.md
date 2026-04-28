# Resume RAG + Job Matching Assignment

This subproject implements Milestone 2 requirements using a local RAG pipeline and job matching engine.

## Files

- `resume_rag.py`
  - Loads resumes using milestone-1 file tools (`fs_tools.py`)
  - Chunks resumes by sections (Education, Experience, Skills, etc.)
  - Generates embeddings (OpenAI)
  - Stores chunks in ChromaDB
  - Extracts metadata: candidate name, skills, experience years, education

- `job_matcher.py`
  - Accepts a job description input
  - Performs semantic retrieval over resume chunks
  - Adds keyword overlap for critical skills (hybrid search)
  - Applies must-have filter for minimum years of experience when specified
  - Returns scored top matches (0-100) with reasoning and relevant excerpts

## Setup

From workspace root:

```powershell
cd assignment_rag_job_matching
pip install -r requirements.txt
```

Ensure root `.env` contains:

```env
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
```

## Run

### 1) Build RAG index

```powershell
python resume_rag.py
```

### 2) Match job description

```powershell
python job_matcher.py --job-description "Looking for Python ML engineer with 5+ years experience in NLP and AWS" --top-k 10 --reindex
```

Or from a file:

```powershell
python job_matcher.py --job-file ../sample_data/job_description.txt --top-k 10
```

## One-Command Demo (for video)

This runs the full flow (index + match) in one command:

```powershell
python demo_run.py
```

Optional custom inputs:

```powershell
python demo_run.py --job-file sample_job_description.txt --top-k 10
```

## Output Format

The matcher returns JSON in the required structure:

```json
{
  "job_description": "...",
  "top_matches": [
    {
      "candidate_name": "John Doe",
      "resume_path": "...",
      "match_score": 92,
      "matched_skills": ["Python", "Machine Learning"],
      "relevant_excerpts": ["..."],
      "reasoning": "Strong match for ML experience..."
    }
  ]
}
```
