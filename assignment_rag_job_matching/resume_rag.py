from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
import chromadb
from openai import OpenAI


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Prefer MCP-based filesystem calls when MCP_URL is set; otherwise fall back to local fs_tools
MCP_URL = None
try:
    import os

    MCP_URL = os.environ.get("MCP_URL")
except Exception:
    MCP_URL = None

if MCP_URL:
    try:
        from mcp_client import call_rpc as _call_rpc

        def list_files(directory: str, extension: str | None = None):
            params = {"directory": directory}
            if extension:
                params["extension"] = extension
            resp = _call_rpc(MCP_URL, "list_files", params)
            return resp.get("result") if isinstance(resp, dict) and "result" in resp else resp

        def read_file(filepath: str):
            resp = _call_rpc(MCP_URL, "read_file", {"filepath": filepath})
            return resp.get("result") if isinstance(resp, dict) and "result" in resp else resp
    except Exception:
        from fs_tools import list_files, read_file
else:
    from fs_tools import list_files, read_file


SECTION_HEADERS = {
    "summary",
    "profile",
    "experience",
    "work experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "achievements",
}

SKILL_KEYWORDS = {
    "python",
    "java",
    "javascript",
    "react",
    "node.js",
    "node",
    "sql",
    "mongodb",
    "postgresql",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "machine learning",
    "nlp",
    "pytorch",
    "scikit-learn",
    "selenium",
    "playwright",
    "fastapi",
    "django",
    "flask",
    "pandas",
    "power bi",
    "excel",
}


@dataclass
class ResumeChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


def default_resumes_dir() -> str:
    return str((ROOT_DIR / "sample_data" / "resumes").resolve())


def default_persist_dir() -> str:
    return str((CURRENT_DIR / "vector_store").resolve())


class ResumeRAG:
    def __init__(
        self,
        resumes_dir: str,
        persist_dir: str,
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "resume_chunks",
        chunk_size: int = 1000,
        chunk_overlap: int = 120,
    ) -> None:
        load_dotenv()
        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.resumes_dir = Path(resumes_dir)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vector_client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.vector_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _extract_sections(self, content: str) -> List[Tuple[str, str]]:
        lines = [line.rstrip() for line in content.splitlines()]
        sections: List[Tuple[str, List[str]]] = []

        current_header = "general"
        current_lines: List[str] = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                current_lines.append(raw_line)
                continue

            line_key = line.lower().strip(":")
            if line_key in SECTION_HEADERS:
                if current_lines:
                    sections.append((current_header, current_lines))
                current_header = line_key
                current_lines = [raw_line]
            else:
                current_lines.append(raw_line)

        if current_lines:
            sections.append((current_header, current_lines))

        return [(header, "\n".join(section_lines).strip()) for header, section_lines in sections if "\n".join(section_lines).strip()]

    def _chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    def _extract_name(self, content: str, fallback_stem: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "@" in stripped:
                continue
            if len(stripped.split()) >= 2 and len(stripped) <= 60:
                if not re.search(r"(summary|skills|experience|education)", stripped, flags=re.IGNORECASE):
                    return stripped
        return fallback_stem.replace("_", " ").replace("-", " ").title()

    def _extract_skills(self, content: str) -> List[str]:
        normalized = content.lower()
        found = [skill for skill in sorted(SKILL_KEYWORDS) if skill in normalized]
        return found

    def _extract_experience_years(self, content: str) -> int:
        direct_matches = re.findall(r"(\d{1,2})\+?\s+years", content, flags=re.IGNORECASE)
        direct_years = [int(match) for match in direct_matches] if direct_matches else []

        range_matches = re.findall(r"(20\d{2})\s*[-–]\s*(20\d{2}|present|current)", content, flags=re.IGNORECASE)
        range_years: List[int] = []
        for start, end in range_matches:
            start_year = int(start)
            end_year = 2026 if end.lower() in {"present", "current"} else int(end)
            if end_year >= start_year:
                range_years.append(end_year - start_year)

        values = direct_years + range_years
        return max(values) if values else 0

    def _extract_education(self, content: str) -> str:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        matches = [
            line
            for line in lines
            if re.search(
                r"(bachelor|master|b\.?tech|m\.?tech|phd|university|college|degree)",
                line,
                flags=re.IGNORECASE,
            )
        ]
        return " | ".join(matches[:3]) if matches else ""

    def extract_metadata(self, content: str, filepath: Path) -> Dict[str, Any]:
        skills = self._extract_skills(content)
        return {
            "candidate_name": self._extract_name(content, filepath.stem),
            "skills": skills,
            "experience_years": self._extract_experience_years(content),
            "education": self._extract_education(content),
            "resume_path": str(filepath.resolve()),
            "file_extension": filepath.suffix.lower(),
        }

    def load_resume_files(self) -> List[Path]:
        files = list_files(str(self.resumes_dir))
        allowed = {".txt", ".pdf", ".docx"}
        paths = [Path(item["path"]) for item in files if item.get("extension", "").lower() in allowed]
        return sorted(paths)

    def build_chunks(self) -> List[ResumeChunk]:
        chunks: List[ResumeChunk] = []
        resume_files = self.load_resume_files()

        for file_path in resume_files:
            read_result = read_file(str(file_path))
            if not read_result.get("success"):
                continue

            content = read_result.get("content", "")
            if not content.strip():
                continue

            resume_meta = self.extract_metadata(content, file_path)
            sections = self._extract_sections(content)

            section_counter = 0
            for section_name, section_text in sections:
                for piece in self._chunk_text(section_text):
                    section_counter += 1
                    chunk_id = f"{file_path.stem}__{section_name}__{section_counter}"
                    chunk_meta = {
                        "candidate_name": resume_meta["candidate_name"],
                        "resume_path": resume_meta["resume_path"],
                        "file_extension": resume_meta["file_extension"],
                        "section": section_name,
                        "skills_csv": ",".join(resume_meta["skills"]),
                        "experience_years": int(resume_meta["experience_years"]),
                        "education": resume_meta["education"] or "",
                    }
                    chunks.append(ResumeChunk(chunk_id=chunk_id, text=piece, metadata=chunk_meta))

        return chunks

    def _embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.embedding_model, input=batch)
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    def rebuild_index(self) -> Dict[str, Any]:
        chunks = self.build_chunks()
        if not chunks:
            return {
                "success": False,
                "error": "No valid resume content found for indexing.",
                "resumes_dir": str(self.resumes_dir.resolve()),
            }

        try:
            self.vector_client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.vector_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings = self._embed_texts(texts)

        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

        unique_resumes = len({meta["resume_path"] for meta in metadatas})
        return {
            "success": True,
            "collection": self.collection_name,
            "resumes_indexed": unique_resumes,
            "chunks_indexed": len(chunks),
            "persist_dir": str(self.persist_dir.resolve()),
        }

    def query_chunks(self, query_text: str, n_results: int = 10) -> Dict[str, Any]:
        query_embedding = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query_text],
        ).data[0].embedding

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume RAG indexing pipeline")
    parser.add_argument("--resumes-dir", default=default_resumes_dir(), help="Directory containing resumes")
    parser.add_argument("--persist-dir", default=default_persist_dir(), help="ChromaDB persistent directory")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Embedding model")
    args = parser.parse_args()

    rag = ResumeRAG(
        resumes_dir=args.resumes_dir,
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
    )
    result = rag.rebuild_index()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
