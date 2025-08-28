import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Optional

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------- Paths & config (no args) --------
BASE_DIR    = Path(__file__).resolve().parent
INDEX_DIR   = BASE_DIR / "vector_db"
OUTPUT_DIR  = BASE_DIR / "output"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM settings (can override via env)
LLM_URL     = os.environ.get("LLM_URL", "http://192.168.0.205:80/v1/chat/completions")
LLM_MODEL   = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
# LLM_API_KEY = os.environ.get("LLM_API_KEY")  # not needed for local LLM

SYSTEM_PROMPT = (
    "You are a research assistant. Give me a summary whether the paper has work related to maternal vaccination?"

)
TIMEOUT = 600  # seconds

# -------- FAISS helpers --------
def load_vector_store(index_dir: Path) -> FAISS:
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"Missing FAISS index at: {index_dir/'index.faiss'}")
    embeddings = HuggingFaceEmbeddings(model=EMBED_MODEL, show_progress=True)
    return FAISS.load_local(
        index_dir,
        embeddings,
        normalize_L2=True,
        allow_dangerous_deserialization=True,
    )

def iter_faiss_entries(vs: FAISS) -> Iterable[Tuple[int, str, Dict[str, Any]]]:
    """
    Yield (faiss_pos, doc_id, payload) for each stored vector entry.
    payload = {"source": str, "content": str, "metadata": dict}
    """
    pos_to_id = vs.index_to_docstore_id  # Dict[int, str]
    docdict   = vs.docstore._dict        # Dict[str, Document]
    for pos in sorted(pos_to_id.keys()):
        doc_id = pos_to_id[pos]
        doc = docdict.get(doc_id)
        if doc is None:
            continue
        content = (doc.page_content or "").strip()
        meta = dict(doc.metadata or {})
        source = meta.get("source", "N/A")
        yield pos, doc_id, {"source": source, "content": content, "metadata": meta}

# -------- LLM caller (supports chat/completions) --------
def call_llm(
    url: str,
    model: str,
    system_prompt: str,
    user_content: str,
    api_key: Optional[str] = None,  # default None so you don't need to pass it
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if url.rstrip("/").endswith("/v1/completions"):
        # Text completions style
        payload = {
            "model": model,
            "prompt": f"System: {system_prompt}\n\nUser:\n{user_content}",
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
        }
    else:
        # Chat completions style
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
        }

    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"LLM error {r.status_code}: {r.text}")
    data = r.json()
    if url.rstrip("/").endswith("/v1/completions"):
        return data["choices"][0]["text"]
    return data["choices"][0]["message"]["content"]

# -------- Main --------
def main() -> None:
    load_dotenv()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"faiss_infer_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    vs = load_vector_store(INDEX_DIR)

    new_file = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["faiss_pos", "doc_id", "source", "text_len", "metadata_json", "llm_output"],
        )
        if new_file:
            writer.writeheader()

        for faiss_pos, doc_id, payload in iter_faiss_entries(vs):
            text = payload["content"]
            meta_json = json.dumps(payload["metadata"], ensure_ascii=False)

            if not text:
                writer.writerow({
                    "faiss_pos": faiss_pos,
                    "doc_id": doc_id,
                    "source": payload["source"],
                    "text_len": 0,
                    "metadata_json": meta_json,
                    "llm_output": "[EMPTY_CONTENT]",
                })
                f.flush()
                continue

            try:
                llm_out = call_llm(
                    url=LLM_URL,
                    model=LLM_MODEL,
                    system_prompt=SYSTEM_PROMPT,
                    user_content=text,
                    # api_key=None  # not needed
                )
            except Exception as e:
                llm_out = f"[LLM_ERROR] {repr(e)}"

            writer.writerow({
                "faiss_pos": faiss_pos,
                "doc_id": doc_id,
                "source": payload["source"],
                "text_len": len(text),
                "metadata_json": meta_json,
                "llm_output": llm_out,
            })
            f.flush()

    print(f"Done. Results -> {out_path}")

if __name__ == "__main__":
    main()
