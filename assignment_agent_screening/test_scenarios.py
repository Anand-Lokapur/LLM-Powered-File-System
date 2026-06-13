"""
Test Scenarios for Screening Agent

5+ conversation flows demonstrating Part A, B, and C capabilities.
"""

try:
    from .matching_agent import ScreeningAgent
except ImportError:
    from matching_agent import ScreeningAgent


def scenario_1_basic_screening():
    """
    Scenario 1: Basic screening flow
    Tests Part A: Agent architecture and initial screening.
    """
    print("=" * 70)
    print("SCENARIO 1: Basic ML Engineer Screening")
    print("=" * 70)

    jd = """
We are hiring a Machine Learning Engineer with 3+ years experience.

Must-have:
- Python programming
- Machine Learning
- Experience with TensorFlow or PyTorch

Nice-to-have:
- NLP experience
- AWS deployment
- MLOps background

Responsibilities: Build ML models, deploy to production, collaborate with product team.
"""

    agent = ScreeningAgent()
    result = agent.run_initial_screening(jd)

    print(f"\nJob Requirements Extracted:")
    if result.get("job_requirements"):
        req = result["job_requirements"]
        print(f"  Must-have: {req.must_have_skills}")
        print(f"  Min years: {req.min_years_experience}")

    print(f"\nCandidates Found: {len(result.get('candidates', []))}")
    print(f"Shortlisted: {len(result.get('shortlist', []))}")

    print(f"\nAgent Response:\n{result.get('agent_response', '')[:300]}")


def scenario_2_candidate_comparison():
    """
    Scenario 2: Compare top candidates
    Tests Part B: Interactive comparison queries.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: Candidate Comparison")
    print("=" * 70)

    jd = "Python engineer with 5+ years and AWS experience needed."
    agent = ScreeningAgent()
    result = agent.run_initial_screening(jd)

    query = "Compare the top 3 candidates for me"
    print(f"\nUser Query: {query}")

    refined_result = agent.process_user_query(result, query)
    print(f"\nAgent Response:\n{refined_result.get('agent_response', '')[:400]}")


def scenario_3_interview_questions():
    """
    Scenario 3: Generate interview questions
    Tests Part C: Tailored screening questions.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: Interview Question Generation")
    print("=" * 70)

    jd = "Full-stack developer with React and Node.js expertise."
    agent = ScreeningAgent()
    result = agent.run_initial_screening(jd)

    query = "Generate interview questions for the top candidate"
    print(f"\nUser Query: {query}")

    refined_result = agent.process_user_query(result, query)
    response = refined_result.get('agent_response', '')
    print(f"\nAgent Response:\n{response[:500]}")


def scenario_4_ranking_explanation():
    """
    Scenario 4: Explain ranking differences
    Tests Part B: Explainability and reasoning.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 4: Ranking Explanation")
    print("=" * 70)

    jd = "Data engineer with SQL and Python. 4+ years required."
    agent = ScreeningAgent()
    result = agent.run_initial_screening(jd)

    query = "Why did the first candidate rank higher than the second?"
    print(f"\nUser Query: {query}")

    refined_result = agent.process_user_query(result, query)
    print(f"\nAgent Response:\n{refined_result.get('agent_response', '')[:400]}")


def scenario_5_iterative_refinement():
    """
    Scenario 5: Iterative refinement with criteria change
    Tests Part B: Adjustment of requirements mid-conversation.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 5: Iterative Refinement")
    print("=" * 70)

    jd = "Backend developer with 3+ years. Java or Python."
    agent = ScreeningAgent()
    result = agent.run_initial_screening(jd)

    print(f"\n1. Initial Search:")
    print(f"   Candidates: {len(result.get('candidates', []))}")
    print(f"   Shortlist: {len(result.get('shortlist', []))}")

    query = "Refine to only candidates with Docker and Kubernetes experience"
    print(f"\n2. User Refinement Query: {query}")

    refined_result = agent.process_user_query(result, query)
    print(f"   Response: {refined_result.get('agent_response', '')[:250]}")


def scenario_6_multi_round_analysis():
    """
    Scenario 6: Multi-round screening with deep analysis
    Tests Part C: Advanced capabilities - multi-round screening.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 6: Multi-Round Screening & Deep Analysis")
    print("=" * 70)

    jd = """
Senior ML Engineer position.
Must: 5+ years Python, PyTorch, AWS
Nice: MLOps, FastAPI, published research
"""

    agent = ScreeningAgent()
    result = agent.run_initial_screening(jd)

    print(f"\nRound 1 - Initial Screening:")
    print(f"  Candidates: {len(result.get('candidates', []))}")
    print(f"  Shortlist: {len(result.get('shortlist', []))}")

    print(f"\nRound 2 - Deep Analysis of Top 3:")
    shortlist = result.get('shortlist', [])
    for idx, candidate in enumerate(shortlist[:3], 1):
        if hasattr(candidate, 'candidate_name'):
            name = candidate.candidate_name
            score = candidate.match_score
            print(f"  {idx}. {name} (Score: {score})")
            if hasattr(candidate, 'strengths') and candidate.strengths:
                print(f"     Strengths: {', '.join(candidate.strengths)}")
            if hasattr(candidate, 'gaps') and candidate.gaps:
                print(f"     Gaps: {', '.join(candidate.gaps)}")


def scenario_7_gap_analysis():
    """
    Scenario 7: Analyze gaps for borderline candidates
    Tests Part C: Explainability - improvement suggestions.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 7: Gap Analysis & Improvement Suggestions")
    print("=" * 70)

    jd = "QA Engineer with Selenium, 3+ years, CI/CD knowledge required."
    agent = ScreeningAgent()
    result = agent.run_initial_screening(jd)

    if result.get('shortlist'):
        candidate = result['shortlist'][0]
        print(f"\nCandidate: {candidate.candidate_name if hasattr(candidate, 'candidate_name') else 'Unknown'}")
        print(f"Match Score: {candidate.match_score if hasattr(candidate, 'match_score') else 'N/A'}")

        if hasattr(candidate, 'gaps'):
            print(f"Gaps: {candidate.gaps}")
        if hasattr(candidate, 'improvement_suggestions'):
            print(f"Suggestions: {candidate.improvement_suggestions}")


def run_all_scenarios():
    """Run all test scenarios."""
    scenarios = [
        scenario_1_basic_screening,
        scenario_2_candidate_comparison,
        scenario_3_interview_questions,
        scenario_4_ranking_explanation,
        scenario_5_iterative_refinement,
        scenario_6_multi_round_analysis,
        scenario_7_gap_analysis,
    ]

    for scenario in scenarios:
        try:
            scenario()
        except Exception as e:
            print(f"\nError in {scenario.__name__}: {str(e)}")
        print()


if __name__ == "__main__":
    print("\nRunning Screening Agent Test Scenarios...")
    print("(Note: Requires OPENAI_API_KEY in .env)\n")

    try:
        run_all_scenarios()
    except Exception as e:
        print(f"Setup error: {str(e)}")
        print("Ensure all dependencies are installed and OpenAI API key is set.")
