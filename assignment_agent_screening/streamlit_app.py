import json
from typing import Any, Dict, Optional

import streamlit as st
from matching_agent import ScreeningAgent


st.set_page_config(page_title="Screening Agent", layout="wide")


def run_screening(agent: ScreeningAgent, jd: str) -> Dict[str, Any]:
    """Run initial screening and return the result dict."""
    return agent.run_initial_screening(jd)


def parse_agent_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse an agent response that may contain JSON payload text."""
    if not raw_text:
        return None

    text = raw_text.strip()

    # Try direct JSON first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Try extracting a JSON object from mixed text.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None

    return None


def render_structured_answer(query: str, raw_answer: str) -> None:
    """Render agent output as readable UI if it is structured JSON."""
    parsed = parse_agent_payload(raw_answer)

    if not parsed:
        st.markdown(raw_answer)
        return

    # Common fields for interview-question style outputs.
    if parsed.get("candidate_name"):
        st.caption(f"Candidate: {parsed.get('candidate_name')}")

    if parsed.get("estimated_level"):
        st.info(f"Estimated level: {parsed.get('estimated_level')}")

    sections = [
        ("Interview questions", parsed.get("interview_questions")),
        ("Technical questions", parsed.get("technical_questions")),
        ("Behavioral questions", parsed.get("behavioral_questions")),
        ("Gap-filling questions", parsed.get("gap_filling_questions")),
        ("Strengths to explore", parsed.get("strengths_to_explore")),
    ]

    rendered_any = False
    for title, items in sections:
        if items:
            rendered_any = True
            st.markdown(f"### {title}")
            for idx, item in enumerate(items, 1):
                st.write(f"{idx}. {item}")

    # Render any other fields as a summary table.
    other_keys = [k for k in parsed.keys() if k not in {
        "candidate_name",
        "estimated_level",
        "interview_questions",
        "technical_questions",
        "behavioral_questions",
        "gap_filling_questions",
        "strengths_to_explore",
    }]
    if other_keys:
        st.markdown("### Additional details")
        st.json({k: parsed[k] for k in other_keys})

    if not rendered_any and not other_keys:
        st.json(parsed)


def main():
    st.title("Candidate Screening Agent — Demo")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Job Description")
        jd = st.text_area("Paste the job description here", height=300)

        if st.button("Run Screening"):
            if not jd.strip():
                st.warning("Please enter a job description before running screening.")
            else:
                agent = ScreeningAgent()
                with st.spinner("Processing JD and searching resumes — this may take a minute..."):
                    result = run_screening(agent, jd)

                st.success("Initial screening finished")
                st.session_state["screening_result"] = result

    with col2:
        st.subheader("Quick Actions")
        if st.button("Load demo JD"):
            demo_jd = st.session_state.get("demo_jd")
            if demo_jd:
                st.experimental_set_query_params()  # noop to keep UI responsive
                st.write("Demo JD loaded into the editor above.")

    # Show results if available
    if "screening_result" in st.session_state:
        result = st.session_state["screening_result"]

        st.markdown("---")
        st.subheader("Processing Trace")
        trace = result.get("trace", [])
        with st.expander("Show processing steps (trace)", expanded=False):
            for line in trace:
                st.text(line)

        st.subheader("Extracted Requirements")
        req = result.get("job_requirements")
        if req:
            st.write(f"**Must-have skills:** {', '.join(req.must_have_skills[:10])}")
            st.write(f"**Min years experience:** {req.min_years_experience}")
            st.write(f"**Nice-to-have:** {', '.join(req.nice_to_have_skills[:10])}")
        else:
            st.info("No requirements parsed.")

        st.subheader("Shortlist")
        shortlist = result.get("shortlist", [])
        if not shortlist:
            st.info("No candidates shortlisted. Check the trace and ensure index is built.")
        else:
            rows = []
            for c in shortlist:
                name = c.candidate_name if hasattr(c, "candidate_name") else c.get("candidate_name")
                score = c.match_score if hasattr(c, "match_score") else c.get("match_score")
                skills = c.matched_skills if hasattr(c, "matched_skills") else c.get("matched_skills", [])
                strengths = c.strengths if hasattr(c, "strengths") else c.get("strengths", [])
                gaps = c.gaps if hasattr(c, "gaps") else c.get("gaps", [])
                rows.append({"name": name, "score": score, "skills": ", ".join(skills[:6]), "strengths": ", ".join(strengths[:4]), "gaps": ", ".join(gaps[:4])})

            st.table(rows)

        st.markdown("---")
        st.subheader("Follow-up Chat / Actions")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        query = st.text_input("Ask the agent about the shortlist (e.g. 'Why did Alice rank above Bob?')")
        cols = st.columns([3, 1])
        with cols[1]:
            ask = st.button("Ask")

        if ask and query:
            agent = ScreeningAgent()
            screening_state = result
            try:
                response = agent.process_user_query(screening_state, query)
                answer = response.get("agent_response", "")
                st.session_state["chat_history"].append((query, answer))
            except Exception as e:
                st.error(f"Error processing query: {e}")

        if st.session_state["chat_history"]:
            st.subheader("Conversation")
            for q, a in reversed(st.session_state["chat_history"]):
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    render_structured_answer(q, a)


if __name__ == "__main__":
    main()
