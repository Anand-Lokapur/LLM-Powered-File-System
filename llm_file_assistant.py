from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from fs_tools import list_files, read_file, search_in_file, write_file


logger = logging.getLogger("llm_file_assistant")


SYSTEM_PROMPT = """
You are a resume file assistant. Use tools to inspect files, search content, and write outputs.
Always call tools when user asks about file content or file operations.
When creating summaries, keep them concise and factual.
""".strip()


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a single TXT/PDF/DOCX file and extract text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "File path to read."}
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory, optionally filtered by extension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory path."},
                    "extension": {
                        "type": ["string", "null"],
                        "description": "Optional file extension filter, e.g. .pdf",
                    },
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": "Search keyword in file content with case-insensitive matching and context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "keyword": {"type": "string"},
                },
                "required": ["filepath", "keyword"],
            },
        },
    },
]


def execute_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Tool call started | tool=%s | args=%s", tool_name, args)

    if tool_name == "read_file":
        result = read_file(**args)
    elif tool_name == "list_files":
        result = {"success": True, "results": list_files(**args)}
    elif tool_name == "write_file":
        result = write_file(**args)
    elif tool_name == "search_in_file":
        result = search_in_file(**args)
    else:
        result = {"success": False, "error": f"Unknown tool: {tool_name}"}

    logger.info(
        "Tool call completed | tool=%s | success=%s | result_keys=%s",
        tool_name,
        result.get("success"),
        list(result.keys()),
    )
    return result


def complete_with_tools(client: OpenAI, messages: List[Dict[str, Any]], model: str) -> str:
    iteration = 0
    while True:
        iteration += 1
        logger.info("LLM request started | iteration=%s", iteration)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        logger.info("LLM response received | iteration=%s", iteration)

        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        logger.info("Tool calls detected | iteration=%s | count=%s", iteration, len(tool_calls))

        if not tool_calls:
            final_text = message.content or ""
            messages.append({"role": "assistant", "content": final_text})
            logger.info("Assistant final response generated | iteration=%s", iteration)
            return final_text

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
                logger.warning(
                    "Tool args parse failed | tool=%s | raw_args=%s",
                    tool_name,
                    tool_call.function.arguments,
                )

            result = execute_tool(tool_name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )


def run_assistant(query: str, model: str = "gpt-4.1-mini") -> str:
    load_dotenv()
    client = OpenAI()
    logger.info("Assistant run started | model=%s | query=%s", model, query)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    return complete_with_tools(client=client, messages=messages, model=model)


def interactive_chat(model: str) -> None:
    print("Resume File Assistant (type 'exit' to quit)")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return

        try:
            answer = run_assistant(query, model=model)
            print(f"\nAssistant: {answer}")
        except Exception as exc:
            print(f"\nAssistant Error: {exc}")


def inline_chatbot(model: str) -> None:
    load_dotenv()
    client = OpenAI()
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Inline Resume Chatbot (type 'exit' to quit)")
    while True:
        question = input("Q> ").strip()
        if question.lower() in {"exit", "quit"}:
            print("A> Goodbye!")
            return
        if not question:
            continue

        try:
            logger.info("Inline question received | question=%s", question)
            messages.append({"role": "user", "content": question})
            answer = complete_with_tools(client=client, messages=messages, model=model)
            print(f"A> {answer}")
        except Exception as exc:
            print(f"A> Error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-powered file system assistant")
    parser.add_argument("query", nargs="?", help="Optional one-shot query.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name.")
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Run inline Q/A chatbot mode in terminal.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Application started | model=%s | one_shot=%s", args.model, bool(args.query))

    if args.inline:
        inline_chatbot(model=args.model)
    elif args.query:
        output = run_assistant(args.query, model=args.model)
        print(output)
    else:
        interactive_chat(model=args.model)


if __name__ == "__main__":
    main()
