# File System AI - Resume Assistant

This project implements file system tools and an LLM-powered assistant that can read, search, list, and write resume-related files.

## Features

### Part A: Core File System Tools (`fs_tools.py`)

- `read_file(filepath: str) -> dict`
  - Supports `.txt`, `.pdf`, `.docx`
  - Extracts text content
  - Returns structured response with content + metadata
  - Graceful error handling

- `list_files(directory: str, extension: str = None) -> list`
  - Lists files recursively in a directory
  - Optional extension filtering (`.pdf`, `.txt`, `.docx`, etc.)
  - Returns metadata (`name`, `path`, `size_bytes`, `modified_at`, `extension`)

- `write_file(filepath: str, content: str) -> dict`
  - Writes text content
  - Creates parent directories if needed
  - Returns success/failure with metadata

- `search_in_file(filepath: str, keyword: str) -> dict`
  - Case-insensitive keyword search
  - Returns matched text + surrounding context + location details

### Part B: LLM Integration (`llm_file_assistant.py`)

- Uses OpenAI tool calling
- LLM can automatically call:
  - `read_file`
  - `list_files`
  - `write_file`
  - `search_in_file`
- Supports interactive chat and one-shot query mode

## Project Structure

```
fileSYstemAI/
├── fs_tools.py
├── llm_file_assistant.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── sample_data/
    └── resumes/
        ├── resume_john_doe.txt
        ├── resume_jane_smith.txt
        ├── resume_rahul_mehta.txt
        ├── resume_emily_clark.txt
        ├── resume_aman_kapoor.txt
        └── resume_sophia_lee.txt
```

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Add your OpenAI API key in `.env`:

```env
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
```

## Usage

### Run in interactive mode

```powershell
python llm_file_assistant.py
```

### Run one-shot query

```powershell
python llm_file_assistant.py "Read all resumes in sample_data/resumes"
```

## MCP-based Filesystem Server

This repository also includes a JSON-RPC 2.0 MCP-style filesystem server and a refactored agent path that can use it.

### Start the server

```powershell
python filesystem_mcp_server.py
```

Available endpoints:
- `POST /rpc` for JSON-RPC calls
- `GET /resources` for resource discovery
- `GET /config` for server configuration

### Run the MCP demo

```powershell
$env:MCP_URL = "http://127.0.0.1:8080/rpc"
python mcp_demo.py
```

### Run the agent directly with MCP

```powershell
$env:MCP_URL = "http://127.0.0.1:8080/rpc"
python -c "from assignment_agent_screening.matching_agent import ScreeningAgent; agent = ScreeningAgent(mcp_url='http://127.0.0.1:8080/rpc'); print(agent.run_initial_screening('Looking for a Python backend engineer with AWS experience')['agent_response'][:800])"
```

### Example JSON-RPC calls

```powershell
python -c "from mcp_client import call_rpc; import json; print(json.dumps(call_rpc('http://127.0.0.1:8080/rpc', 'list_files', {'directory': 'sample_data/resumes'}), indent=2))"
```

```powershell
python -c "from mcp_client import call_rpc; import json; print(json.dumps(call_rpc('http://127.0.0.1:8080/rpc', 'batch_process', {'filepaths': ['sample_data/resumes/resume_john_doe.txt'], 'operation': 'read'}), indent=2))"
```

## Example Queries

- `Read all resumes in the sample_data/resumes folder`
- `Find resumes mentioning Python experience`
- `Create a summary file for sample_data/resumes/resume_john_doe.txt`

## Demo Video (Submission Requirement)

Record a 2-3 minute demo showing:

1. Setup and environment configuration
2. Tool calling in action via `llm_file_assistant.py`
3. Example queries from the assignment
4. Generated output file(s)
