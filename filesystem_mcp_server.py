from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

from pypdf import PdfReader
from docx import Document

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def _get_file_metadata(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
    }


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


class MCPServer:
    def __init__(self):
        self.watches: Dict[str, Dict[str, Any]] = {}
        self.config: Dict[str, Any] = {"poll_interval": 2.0}
        self.executor = ThreadPoolExecutor(max_workers=8)

    # Milestone 1 tools
    def list_files(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        directory = params.get("directory")
        extension = params.get("extension")
        if not directory:
            return []
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        normalized_ext = None
        if extension:
            normalized_ext = extension.lower()
            if not normalized_ext.startswith("."):
                normalized_ext = f".{normalized_ext}"

        results: List[Dict[str, Any]] = []
        for path in sorted(dir_path.rglob("*")):
            if not path.is_file():
                continue
            if normalized_ext and path.suffix.lower() != normalized_ext:
                continue
            results.append(_get_file_metadata(path))
        return results

    def read_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        filepath = params.get("filepath")
        if not filepath:
            return {"success": False, "error": "filepath required"}
        path = Path(filepath)
        if not path.exists() or not path.is_file():
            return {"success": False, "error": f"File not found: {filepath}", "filepath": filepath}
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return {"success": False, "error": f"Unsupported extension {ext}", "filepath": filepath}
        try:
            if ext == ".txt":
                content = _read_txt(path)
            elif ext == ".pdf":
                content = _read_pdf(path)
            else:
                content = _read_docx(path)
            return {"success": True, "filepath": str(path.resolve()), "metadata": _get_file_metadata(path), "content": content}
        except Exception as exc:
            return {"success": False, "error": f"Failed to read: {exc}", "filepath": filepath}

    def write_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        filepath = params.get("filepath")
        content = params.get("content", "")
        if not filepath:
            return {"success": False, "error": "filepath required"}
        path = Path(filepath)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {"success": True, "filepath": str(path.resolve()), "bytes_written": len(content.encode("utf-8")), "metadata": _get_file_metadata(path)}
        except Exception as exc:
            return {"success": False, "error": f"Failed to write: {exc}", "filepath": filepath}

    def search_in_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        filepath = params.get("filepath")
        keyword = params.get("keyword", "")
        if not keyword:
            return {"success": False, "error": "keyword required"}
        read = self.read_file({"filepath": filepath})
        if not read.get("success"):
            return {"success": False, "error": read.get("error"), "matches": [], "matches_count": 0}
        content = read.get("content", "")
        matches = []
        import re

        for match in re.finditer(re.escape(keyword), content, flags=re.IGNORECASE):
            start, end = match.start(), match.end()
            context_start = max(0, start - 60)
            context_end = min(len(content), end + 60)
            line_number = content.count("\n", 0, start) + 1
            matches.append({"match": content[start:end], "start_index": start, "end_index": end, "line_number": line_number, "context": content[context_start:context_end].strip()})
        return {"success": True, "filepath": filepath, "keyword": keyword, "matches_count": len(matches), "matches": matches}

    # MCP-specific capabilities
    def batch_process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        files = params.get("filepaths", [])
        operation = params.get("operation", "read")
        if not isinstance(files, list):
            return {"success": False, "error": "filepaths must be a list"}

        futures = []
        results = {}

        def _do(filepath: str):
            if operation == "read":
                return self.read_file({"filepath": filepath})
            elif operation == "metadata":
                p = Path(filepath)
                if p.exists() and p.is_file():
                    return {"success": True, "metadata": _get_file_metadata(p), "filepath": str(p.resolve())}
                else:
                    return {"success": False, "error": "not found", "filepath": filepath}
            else:
                return {"success": False, "error": f"unsupported operation {operation}"}

        for f in files:
            futures.append(self.executor.submit(_do, f))

        for idx, fut in enumerate(futures):
            results[files[idx]] = fut.result()

        return {"success": True, "results": results}

    def watch_directory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        directory = params.get("directory")
        if not directory:
            return {"success": False, "error": "directory required"}
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return {"success": False, "error": "directory not found"}

        watch_id = str(uuid.uuid4())
        state = {"directory": directory, "events": [], "running": True}
        self.watches[watch_id] = state

        def _watch():
            seen = set(p.resolve() for p in dir_path.rglob("*") if p.is_file())
            while state["running"]:
                time.sleep(self.config.get("poll_interval", 2.0))
                current = set(p.resolve() for p in dir_path.rglob("*") if p.is_file())
                added = current - seen
                for p in added:
                    ev = {"type": "created", "path": str(p), "metadata": _get_file_metadata(Path(p))}
                    state["events"].append(ev)
                seen = current

        t = threading.Thread(target=_watch, daemon=True)
        t.start()

        return {"success": True, "watch_id": watch_id}

    def get_watch_events(self, params: Dict[str, Any]) -> Dict[str, Any]:
        watch_id = params.get("watch_id")
        if not watch_id or watch_id not in self.watches:
            return {"success": False, "error": "watch_id not found"}
        state = self.watches[watch_id]
        events = list(state.get("events", []))
        state["events"] = []
        return {"success": True, "events": events}


SERVER = MCPServer()


class JSONRPCHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: Dict[str, Any], status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/resources"):
            methods = {
                "list_files": "List files in directory",
                "read_file": "Read file content",
                "write_file": "Write content to file",
                "search_in_file": "Search keyword in file",
                "batch_process": "Process multiple files concurrently",
                "watch_directory": "Start watching directory for new files",
                "get_watch_events": "Fetch watch events",
            }
            self._send_json({"jsonrpc": "2.0", "resources": methods})
            return
        if self.path.startswith("/config"):
            self._send_json({"jsonrpc": "2.0", "config": SERVER.config})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/rpc":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            req = json.loads(raw)
        except Exception:
            return self._send_json({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}, status=400)

        # JSON-RPC 2.0 basic validation
        jsonrpc = req.get("jsonrpc")
        method = req.get("method")
        params = req.get("params", {})
        req_id = req.get("id")

        if jsonrpc != "2.0" or not method:
            return self._send_json({"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": req_id}, status=400)

        # Dispatch
        try:
            if not hasattr(SERVER, method):
                raise AttributeError(f"Method {method} not found")
            func = getattr(SERVER, method)
            result = func(params if isinstance(params, dict) else {})
            resp = {"jsonrpc": "2.0", "result": result, "id": req_id}
            return self._send_json(resp)
        except AttributeError as ae:
            return self._send_json({"jsonrpc": "2.0", "error": {"code": -32601, "message": str(ae)}, "id": req_id}, status=404)
        except Exception as exc:
            return self._send_json({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(exc)}, "id": req_id}, status=500)


def run(host: str = "127.0.0.1", port: int = 8080):
    server = HTTPServer((host, port), JSONRPCHandler)
    print(f"MCP filesystem server running at http://{host}:{port}/rpc")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down MCP server...")
        server.shutdown()


if __name__ == "__main__":
    run()
