"""
Programmatic demo of the Screening Agent.

Usage: python demo_agent.py
"""

import json
from matching_agent import ScreeningAgent


SAMPLE_JD = """
We are seeking a Senior Machine Learning Engineer to join our team.

Position: Senior ML Engineer
Location: Remote

Must-have Requirements:
- 5+ years of software engineering experience
- Expert-level Python programming
- Hands-on Machine Learning and NLP experience
- AWS or Cloud platform experience
- Experience with ML model deployment

Nice-to-have:
- FastAPI or Django framework experience
- MLOps best practices
- Published research or open-source contributions
- Leadership experience

Responsibilities:
- Design and build ML-powered features
- Deploy models to production
- Collaborate with product and data teams
- Optimize model performance
- Mentor junior engineers
- Participate in code reviews

Salary: $180K - $250K
Benefits: 401k, health insurance, remote flexibility, L&D budget
"""


def demo_initial_screening():
    """Demo 1: Run initial screening."""
    print("=" * 80)
    print("DEMO 1: Initial Screening Pipeline (Part A)")
    print("=" * 80)

    agent = ScreeningAgent()

    print("\n📋 Starting candidate screening...\n")
    result = agent.run_initial_screening(SAMPLE_JD)

    print("✅ Initial Screening Complete!\n")

    if result.get("job_requirements"):
        req = result["job_requirements"]
        print("Extracted Requirements:")
        print(f"  • Must-have skills: {', '.join(req.must_have_skills[:5])}")
        print(f"  • Min experience: {req.min_years_experience} years")
        print(f"  • Nice-to-have: {', '.join(req.nice_to_have_skills[:3])}")

    print(f"\nCandidates Found: {len(result.get('candidates', []))}")
    print(f"Shortlisted: {len(result.get('shortlist', []))}")

    print("\nTop 3 Candidates:")
    for idx, candidate in enumerate(result.get("shortlist", [])[:3], 1):
        name = candidate.candidate_name if hasattr(candidate, 'candidate_name') else candidate.get('candidate_name')
        score = candidate.match_score if hasattr(candidate, 'match_score') else candidate.get('match_score')
        print(f"  {idx}. {name} - Score: {score:.1f}/100")

    return result


def demo_interactive_queries(screening_result):
    """Demo 2: Interactive queries (Part B)."""
    print("\n" + "=" * 80)
    print("DEMO 2: Interactive Queries & Refinement (Part B)")
    print("=" * 80)

    agent = ScreeningAgent()

    queries = [
        "Compare the top 3 candidates",
        "Why did the first candidate rank highest?",
        "Generate interview questions for the top candidate",
        "Refine search to only candidates with AWS experience",
    ]

    print("\n🔍 Processing interactive queries...\n")

    for query in queries:
        print(f"User: {query}")
        try:
            result = agent.process_user_query(screening_result, query)
            response = result.get("agent_response", "")
            print(f"Agent: {response[:250]}...")
            print()
        except Exception as e:
            print(f"Agent: Error processing query - {str(e)}\n")


def demo_deep_analysis(screening_result):
    """Demo 3: Deep analysis (Part C)."""
    print("\n" + "=" * 80)
    print("DEMO 3: Advanced Analysis & Recommendations (Part C)")
    print("=" * 80)

    print("\n📊 Running deep analysis on shortlisted candidates...\n")

    shortlist = screening_result.get("shortlist", [])

    for idx, candidate in enumerate(shortlist[:3], 1):
        name = candidate.candidate_name if hasattr(candidate, 'candidate_name') else candidate.get('candidate_name')
        score = candidate.match_score if hasattr(candidate, 'match_score') else candidate.get('match_score')
        strengths = candidate.strengths if hasattr(candidate, 'strengths') else candidate.get('strengths', [])
        gaps = candidate.gaps if hasattr(candidate, 'gaps') else candidate.get('gaps', [])

        print(f"Candidate {idx}: {name}")
        print(f"  Match Score: {score}/100")
        if strengths:
            print(f"  Strengths: {', '.join(strengths[:2])}")
        if gaps:
            print(f"  Gaps: {', '.join(gaps[:2])}")
        print()


def demo_full_workflow():
    """Run complete demonstration."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "SCREENING AGENT - COMPREHENSIVE DEMO" + " " * 29 + "║")
    print("║" + " " * 78 + "║")
    print("║" + " " * 10 + "Part A: Multi-stage Agent Architecture (LangGraph)" + " " * 18 + "║")
    print("║" + " " * 10 + "Part B: Interactive Queries & Refinement (LangChain)" + " " * 16 + "║")
    print("║" + " " * 10 + "Part C: Advanced Analysis & Explainability" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")

    print("\n🚀 Starting Multi-Stage Candidate Screening System\n")

    print("Job Description Length:", len(SAMPLE_JD), "chars")
    print("Position: Senior ML Engineer\n")

    screening_result = demo_initial_screening()

    try:
        demo_interactive_queries(screening_result)
    except Exception as e:
        print(f"⚠️  Interactive demo error: {str(e)}")

    try:
        demo_deep_analysis(screening_result)
    except Exception as e:
        print(f"⚠️  Analysis demo error: {str(e)}")

    print("=" * 80)
    print("✨ Demo Complete!")
    print("=" * 80)
    print("\n📚 For more information, see README.md")
    print("🎯 For test scenarios, run: python test_scenarios.py")
    print("💬 For interactive mode, run: python agent_cli.py\n")


if __name__ == "__main__":
    try:
        demo_full_workflow()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n⚠️  Make sure to:")
        print("   1. Set OPENAI_API_KEY in .env")
        print("   2. Run: pip install -r requirements.txt")
        print("   3. Ensure RAG vector store is built (from Milestone 2)")
