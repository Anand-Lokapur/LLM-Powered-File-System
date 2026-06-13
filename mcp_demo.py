from __future__ import annotations

import json
import os
from pathlib import Path

from mcp_client import call_rpc, fetch_json
from assignment_agent_screening.matching_agent import ScreeningAgent


MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8080/rpc")


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def show_resources() -> None:
    print_header("MCP RESOURCE DISCOVERY")
    result = call_rpc(MCP_URL, "list_files", {"directory": str(Path("sample_data/resumes").resolve())})
    if "error" in result:
        print("Server error:", result["error"])
        return
    print("Sample list_files response:")
    print(json.dumps(result, indent=2)[:2000])

    discovery = fetch_json("http://127.0.0.1:8080/resources")
    if "error" not in discovery:
        print("\nGET /resources:")
        print(json.dumps(discovery, indent=2)[:2000])

    config = fetch_json("http://127.0.0.1:8080/config")
    if "error" not in config:
        print("\nGET /config:")
        print(json.dumps(config, indent=2)[:1000])


def show_batch_process() -> None:
    print_header("MCP BATCH PROCESS")
    resumes_dir = Path("sample_data/resumes").resolve()
    listing = call_rpc(MCP_URL, "list_files", {"directory": str(resumes_dir), "extension": ".txt"})
    files = []
    if isinstance(listing, dict) and "result" in listing:
        files = [item["path"] for item in listing["result"][:3]]

    if not files:
        print("No files returned from list_files.")
        return

    batch = call_rpc(MCP_URL, "batch_process", {"filepaths": files, "operation": "read"})
    print(json.dumps(batch, indent=2)[:4000])


def run_agent_demo() -> None:
    print_header("AGENT ↔ MCP END-TO-END DEMO")
    agent = ScreeningAgent(mcp_url=MCP_URL)
    job_description = (
        "Looking for a Python backend engineer with AWS experience, "
        "5+ years of software engineering, and exposure to FastAPI or Django."
    )
    result = agent.run_initial_screening(job_description)
    print("Agent response preview:")
    print(result.get("agent_response", "")[:1200])
    print("\nShortlist size:", len(result.get("shortlist", [])))


def main() -> None:
    print_header("MCP DEMO START")
    print("MCP_URL:", MCP_URL)
    print("If the server is not running, start it with: python filesystem_mcp_server.py")

    show_resources()
    show_batch_process()
    run_agent_demo()


if __name__ == "__main__":
    main()
