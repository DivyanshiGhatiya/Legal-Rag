import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.config import Settings
import ollama


DEFAULT_COLLECTION_NAME = "legal_rag_chunks"
DEFAULT_PERSIST_DIR = os.path.join("data", "chroma")
EMBED_MODEL = "nomic-embed-text"


class LocalVectorStoreError(RuntimeError):
    """Raised when the local embedding/vector store flow fails."""


def compute_file_hash(file_bytes: bytes) -> str:
    """Return a stable SHA-256 fingerprint for a file."""
    return hashlib.sha256(file_bytes).hexdigest()


def _normalize_chunk_inputs(chunks: Sequence[Any]) -> List[Dict[str, Any]]:
    """Normalize a list of chunks into a list of dicts with chunk_id and text."""
    normalized: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("content") or ""
            chunk_id = chunk.get("chunk_id") or f"chunk_{index:03d}"
        else:
            text = str(chunk)
            chunk_id = f"chunk_{index:03d}"

        if not text:
            continue

        normalized.append({"chunk_id": chunk_id, "text": text})
    return normalized


def _manifest_path(persist_dir: str) -> str:
    os.makedirs(persist_dir, exist_ok=True)
    return os.path.join(persist_dir, "manifest.json")


def _load_manifest(persist_dir: str, manifest_path: Optional[str] = None) -> Dict[str, Any]:
    path = manifest_path or _manifest_path(persist_dir)
    if not os.path.exists(path):
        return {"documents": []}

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "documents" in data:
        return data
    if isinstance(data, list):
        return {"documents": data}
    return {"documents": []}


def _save_manifest(manifest: Dict[str, Any], persist_dir: str, manifest_path: Optional[str] = None) -> str:
    path = manifest_path or _manifest_path(persist_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return path


def _get_ollama_client() -> Any:
    """Create a client pointed at the local Ollama server explicitly."""
    return ollama.Client(host="http://127.0.0.1:11434")


def document_already_embedded(
    document_name: Optional[str],
    document_hash: Optional[str],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    manifest_path: Optional[str] = None,
) -> bool:
    """Return True if a document with the same name or hash was already embedded."""
    manifest = _load_manifest(persist_dir, manifest_path)
    for entry in manifest.get("documents", []):
        if document_hash and entry.get("source_hash") == document_hash:
            return True
        if document_name and entry.get("source_name") == document_name:
            return True
    return False

def _embed_texts(texts: Sequence[str]) -> List[List[float]]:
        if not texts:
         return []

        try:
           client = _get_ollama_client()
           response = client.embed(model=EMBED_MODEL, input=list(texts))
        except Exception as exc:
           raise LocalVectorStoreError(f"Unable to contact Ollama at the local API: {exc}") from exc

        if isinstance(response, dict):
            embeddings = response.get("embeddings")
        else:
         embeddings = getattr(response, "embeddings", None)

        if not embeddings:
         raise LocalVectorStoreError("Ollama returned no embeddings.")
        
         return [list(map(float, emb)) for emb in embeddings]
def embed_chunks(
    chunks: Sequence[Any],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    document_name: Optional[str] = None,
    document_hash: Optional[str] = None,
    manifest_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Embed a list of chunks and store them in a local Chroma collection."""
    normalized = _normalize_chunk_inputs(chunks)
    if not normalized:
        return []

    if document_already_embedded(document_name, document_hash, collection_name, persist_dir, manifest_path):
        print(f"Document {document_name or document_hash} already embedded; skipping re-embedding.")
        return []

    documents = [item["text"] for item in normalized]
    ids = [item["chunk_id"] for item in normalized]
    prefixed = [f"search_document: {text}" for text in documents]

    print(f"Embedding {len(normalized)} chunks with {EMBED_MODEL}...")
    embeddings = _embed_texts(prefixed)

    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
    collection = client.get_or_create_collection(name=collection_name)

    for idx, (chunk_id, text, embedding) in enumerate(zip(ids, documents, embeddings), start=1):
        print(f"Embedded chunk {idx}/{len(normalized)}")
        metadata = {
            "source_name": document_name or "",
            "source_hash": document_hash or "",
            "collection_name": collection_name,
        }
        collection.add(documents=[text], embeddings=[embedding], ids=[chunk_id], metadatas=[metadata])

    manifest = _load_manifest(persist_dir, manifest_path)
    entry = {
        "source_name": document_name or "",
        "source_hash": document_hash or "",
        "chunk_ids": ids,
        "chunk_count": len(normalized),
        "records": [
            {
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "embedding": embedding,
            }
            for item, embedding in zip(normalized, embeddings)
        ],
    }
    manifest["documents"] = [
        existing
        for existing in manifest.get("documents", [])
        if not (
            (document_hash and existing.get("source_hash") == document_hash)
            or (document_name and existing.get("source_name") == document_name and not document_hash)
        )
    ] + [entry]
    _save_manifest(manifest, persist_dir, manifest_path)

    return entry["records"]
def save_index_manifest(records: Sequence[Dict[str, Any]], output_path: str = "data/chroma_manifest.json") -> str:
    """Persist a small manifest describing the embedded chunks."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    manifest = {"documents": list(records)}
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return output_path
# ----------------------------------------------------------------------

#----------------------------------------------------------------------
# 5. Retrieval (top-k similarity search)
#----------------------------------------------------------------------
def retrieve_chunks(query, collection_name, persist_dir, top_k=5, document_filter=None):
    """Embed the query and return top-k similar chunks from ChromaDB."""
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError("ChromaDB dir not found: %s" % persist_dir)
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
    collection = client.get_or_create_collection(name=collection_name)
    query_vec = _embed_texts(["search_query: " + query])[0]
    where = None
    if document_filter:
        where = {"source_name": document_filter}
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    retrieved = []
    if not results or not results.get("ids") or not results["ids"][0]:
        return retrieved
    for i, chunk_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i] if results.get("metadatas") else {}
        retrieved.append({
            "chunk_id": chunk_id,
            "text": results["documents"][0][i] if results.get("documents") else "",
            "score": 1.0 - float(results["distances"][0][i]),
            "document_name": meta.get("source_name", "?"),
            "hierarchy": meta.get("hierarchy", "(none)"),
            "chunk_length": meta.get("chunk_length", 0),
        })
    return retrieved
