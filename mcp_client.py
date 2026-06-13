from __future__ import annotations

import json
from typing import Any, Dict
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def call_rpc(url: str, method: str, params: dict | None = None, req_id: str | None = None) -> dict:
    payload = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": req_id or "1"}
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as e:
        return {"error": f"HTTPError: {e.code} {e.reason}"}
    except URLError as e:
        return {"error": f"URLError: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def fetch_json(url: str) -> dict:
    req = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as e:
        return {"error": f"HTTPError: {e.code} {e.reason}"}
    except URLError as e:
        return {"error": f"URLError: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}
