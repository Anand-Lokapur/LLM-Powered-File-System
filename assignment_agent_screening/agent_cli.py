from __future__ import annotations

import argparse
import json
from typing import Any

from matching_agent import ScreeningAgent


DEMO_JOB_DESCRIPTION = """
We are seeking a Senior Python Machine Learning Engineer with hands-on experience.

Must-have:
- 5+ years of software engineering
- Strong Python skills
- Machine Learning and NLP experience
- AWS or Cloud experience

Nice-to-have:
- FastAPI or Django
- MLOps background
- Published research

Responsibilities include building ML models and deploying them to production.
"""

DEMO_QUERIES = [
    "show shortlist",
    "compare the top 3 candidates",
    "why did the first candidate rank higher than the second?",
    "generate interview questions for the top candidate",
    "full-analysis",
]


class ScreeningCLI:
    """
    Part B: Interactive CLI for agent-based candidate screening.
    Handles conversational queries and iterative refinement.
    """

    def __init__(self):
        self.agent = ScreeningAgent()
        self.session_state: dict[str, Any] = {}
        self.is_running = True

    def print_section(self, title: str) -> None:
        """Print a formatted section header."""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")

    def print_response(self, response: str) -> None:
        """Print agent response with formatting."""
        print("Agent:", response[:500] if len(response) > 500 else response)
        if len(response) > 500:
            print(f"... (truncated, {len(response)} chars total)")

    def print_trace(self, trace: list[Any], title: str = "SCREENING STEPS") -> None:
        """Print step-by-step execution trace."""
        if not trace:
            return

        self.print_section(title)
        for idx, step in enumerate(trace, 1):
            print(f"{idx:2}. {step}")

    def display_shortlist(self, shortlist: list[Any]) -> None:
        """Display candidate shortlist in formatted table."""
        if not shortlist:
            print("No candidates in shortlist.")
            return

        print("\nShortlist:")
        print("-" * 90)
        for idx, candidate in enumerate(shortlist[:10], 1):
            name = candidate.candidate_name if hasattr(candidate, 'candidate_name') else candidate.get('candidate_name', 'N/A')
            score = candidate.match_score if hasattr(candidate, 'match_score') else candidate.get('match_score', 0)
            skills = candidate.matched_skills if hasattr(candidate, 'matched_skills') else candidate.get('matched_skills', [])
            skill_str = ", ".join(skills[:3]) if skills else "N/A"
            print(f"{idx:2}. {name:25} | Score: {score:6.1f} | Skills: {skill_str}")
        print("-" * 90)

    def handle_initial_input(self, demo_mode: bool = False) -> None:
        """Handle initial job description input."""
        self.print_section("CANDIDATE SCREENING AGENT")
        print("Welcome to the Screening Agent!")
        print("Part A: Multi-stage agent orchestration")
        print("Part B: Interactive refinement and explanations")
        print("Part C: Deep analysis with hire/no-hire recommendations\n")

        if demo_mode:
            jd = DEMO_JOB_DESCRIPTION.strip()
            print("Using demo mode job description...\n")
        else:
            jd = input("Paste job description (or press Enter for demo): ").strip()

            if not jd:
                jd = DEMO_JOB_DESCRIPTION.strip()
                print("Using demo job description...")

        self.print_section("RUNNING INITIAL SCREENING")
        print(f"Processing JD ({len(jd)} chars)...\n")

        result = self.agent.run_initial_screening(jd)
        self.session_state = result

        self.print_response(result.get("agent_response", ""))
        self.print_trace(result.get("trace", []), title="SCREENING STEPS")
        self.display_shortlist(result.get("shortlist", []))

    def display_help(self) -> None:
        """Display available commands."""
        help_text = """
Available Commands:
  compare [n]        - Compare top N candidates side-by-side (e.g., 'compare 3')
  questions [name]   - Generate interview questions for candidate (e.g., 'questions John Doe')
  explain [names]    - Explain why one candidate ranked higher (e.g., 'explain John vs Jane')
  refine [criteria]  - Refine search criteria and re-rank
  report             - Display full screening report
  shortlist          - Show current shortlist
  full-analysis      - Run full Part C analysis on top candidates
  help               - Show this help message
  quit/exit          - Exit the agent

Examples:
  "Find me candidates with React and 3+ years"
  "Compare the top 3 matches"
  "Generate interview questions for the top candidate"
  "Why did John rank higher than Jane?"
"""
        print(help_text)

    def handle_user_query(self, query: str) -> None:
        """Process user query through agent."""
        if query.lower() in {"help", "?"}:
            self.display_help()
            return

        if query.lower() in {"quit", "exit"}:
            self.is_running = False
            print(
                "Thank you for using the Screening Agent. Goodbye!"
            )
            return

        if query.lower() == "report":
            self._show_report()
            return

        if query.lower() == "shortlist":
            self.display_shortlist(self.session_state.get("shortlist", []))
            return

        if query.lower() == "full-analysis":
            self._run_full_analysis()
            return

        result = self.agent.process_user_query(self.session_state, query)
        self.session_state.update(result)

        self.print_response(result.get("agent_response", ""))
        self.print_trace(result.get("trace", []), title="SCREENING STEPS")

    def _show_report(self) -> None:
        """Display full screening report."""
        self.print_section("SCREENING REPORT")
        shortlist = self.session_state.get("shortlist", [])
        candidates = self.session_state.get("candidates", [])

        print(f"Total Candidates Searched: {len(candidates)}")
        print(f"Shortlisted: {len(shortlist)}\n")

        for idx, candidate in enumerate(shortlist[:5], 1):
            name = candidate.candidate_name if hasattr(candidate, 'candidate_name') else candidate.get('candidate_name')
            score = candidate.match_score if hasattr(candidate, 'match_score') else candidate.get('match_score')
            skills = candidate.matched_skills if hasattr(candidate, 'matched_skills') else candidate.get('matched_skills', [])
            strengths = candidate.strengths if hasattr(candidate, 'strengths') else candidate.get('strengths', [])
            gaps = candidate.gaps if hasattr(candidate, 'gaps') else candidate.get('gaps', [])

            print(f"\n{idx}. {name}")
            print(f"   Match Score: {score}/100")
            print(f"   Key Skills: {', '.join(skills[:5])}")
            if strengths:
                print(f"   Strengths: {', '.join(strengths)}")
            if gaps:
                print(f"   Gaps: {', '.join(gaps)}")

    def _run_full_analysis(self) -> None:
        """Part C: Run deep analysis on top candidates."""
        self.print_section("FULL CANDIDATE ANALYSIS (Part C)")
        shortlist = self.session_state.get("shortlist", [])

        if not shortlist:
            print("No shortlist to analyze.")
            return

        print(f"Running deep analysis on top {min(3, len(shortlist))} candidates...\n")

        for candidate in shortlist[:3]:
            if hasattr(candidate, 'candidate_name'):
                name = candidate.candidate_name
            else:
                name = candidate.get('candidate_name', 'Unknown')
            print(f"\n--- Analyzing {name} ---")

    def run_interactive_loop(self, demo_mode: bool = False, auto_demo: bool = False) -> None:
        """Main interactive loop for Part B."""
        self.handle_initial_input(demo_mode=demo_mode)

        if auto_demo:
            self.print_section("DEMO SHOWCASE")
            print("Running scripted demo queries...\n")
            for query in DEMO_QUERIES:
                print(f"\nYou: {query}")
                if query == "show shortlist":
                    self.display_shortlist(self.session_state.get("shortlist", []))
                    continue
                self.handle_user_query(query)

        self.print_section("INTERACTIVE MODE")
        print("You can now ask questions, refine criteria, or request detailed analysis.")
        print("Type 'help' for available commands.\n")

        while self.is_running:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue

                self.handle_user_query(user_input)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                self.is_running = False
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Type 'help' for available commands.")


def main():
    """Entry point for CLI."""
    parser = argparse.ArgumentParser(description="Agent-based candidate screening CLI")
    parser.add_argument("--demo", action="store_true", help="Run with built-in demo job description")
    parser.add_argument("--auto-demo", action="store_true", help="Run scripted demo queries after screening")
    args = parser.parse_args()

    cli = ScreeningCLI()
    cli.run_interactive_loop(demo_mode=args.demo, auto_demo=args.auto_demo)


if __name__ == "__main__":
    main()
