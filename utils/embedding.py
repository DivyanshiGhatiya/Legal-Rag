import os
import json
import hashlib
import chromadb
from chromadb.config import Settings
import ollama

DEFAULT_COLLECTION_NAME = "legal_rag_chunks"
DEFAULT_PERSIST_DIR = os.path.join("data", "chroma")
EMBED_MODEL = "nomic-embed-text"


def compute_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def embed_texts(texts, prefix="search_document: "):
    if not texts:
        return []
    client = ollama.Client(host="http://127.0.0.1:11434")
    prefixed = [prefix + t for t in texts]
    try:
        response = client.embed(model=EMBED_MODEL, input=prefixed)
    except Exception as exc:
        raise RuntimeError(
            f"Ollama embedding failed for model '{EMBED_MODEL}'. "
            "Make sure Ollama is installed correctly, running, and the model is available. "
            f"Original error: {exc}"
        ) from exc
    embeddings = response.embeddings if hasattr(response, "embeddings") else response.get("embeddings", [])
    if not embeddings:
        raise RuntimeError(
            f"Ollama returned no embeddings for model '{EMBED_MODEL}'. "
            "Run 'ollama pull nomic-embed-text' and try again."
        )
    return [list(map(float, e)) for e in embeddings]


def embed_query(query):
    return embed_texts([query], prefix="search_query: ")[0]


def document_already_embedded(document_name, document_hash, collection_name=DEFAULT_COLLECTION_NAME, persist_dir=DEFAULT_PERSIST_DIR):
    manifest_path = os.path.join(persist_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data.get("documents", []) if isinstance(data, dict) else data
        if document_hash:
            return any(
                entry.get("source_hash") == document_hash
                and entry.get("source_name") == document_name
                for entry in docs
            )
        for entry in docs:
            if document_name and entry.get("source_name") == document_name:
                return True
        return False
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to read indexed-document manifest '{manifest_path}'."
        ) from exc


def get_chroma_client(persist_dir=DEFAULT_PERSIST_DIR):
    return chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))


def get_collection(collection_name=DEFAULT_COLLECTION_NAME, persist_dir=DEFAULT_PERSIST_DIR):
    client = get_chroma_client(persist_dir)
    return client.get_or_create_collection(name=collection_name)


def embed_chunks(chunks, collection_name=DEFAULT_COLLECTION_NAME, persist_dir=DEFAULT_PERSIST_DIR, document_name=None, document_hash=None):
    normalized = []
    for i, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("content") or ""
            cid = chunk.get("chunk_id") or "chunk_{:03d}".format(i)
            chunk_meta = chunk.get("metadata", {}) or {}
            section = chunk.get("section", "")
            subsection = chunk.get("subsection", "")
        else:
            text = str(chunk)
            cid = "chunk_{:03d}".format(i)
            chunk_meta = {}
            section = ""
            subsection = ""
        if text:
            normalized.append({
                "chunk_id": cid,
                "text": text,
                "section": section,
                "subsection": subsection,
                "hierarchy": chunk_meta.get("hierarchy", []),
                "hierarchy_depth": chunk_meta.get("hierarchy_depth", 0),
                "section_number": chunk_meta.get("section_number") or "",
            })

    if not normalized:
        return []
    if document_already_embedded(document_name, document_hash, collection_name, persist_dir):
        return []

    documents = [n["text"] for n in normalized]
    ids = [n["chunk_id"] for n in normalized]
    embeddings = embed_texts(documents, prefix="search_document: ")

    collection = get_collection(collection_name, persist_dir)

    for n, emb in zip(normalized, embeddings):
        meta = {
            "source_name": document_name or "",
            "source_hash": document_hash or "",
            "section": n["section"] or "",
            "subsection": n["subsection"] or "",
            "hierarchy": " > ".join(n["hierarchy"]) if n["hierarchy"] else "",
            "hierarchy_depth": n["hierarchy_depth"],
            "section_number": n["section_number"],
            "chunk_length": len(n["text"]),
        }
        collection.add(documents=[n["text"]], embeddings=[emb], ids=[n["chunk_id"]], metadatas=[meta])

    os.makedirs(persist_dir, exist_ok=True)
    manifest_path = os.path.join(persist_dir, "manifest.json")
    manifest = {"documents": []}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                old = json.load(f)
                manifest = old if isinstance(old, dict) else {"documents": old}
        except Exception:
            pass

    manifest["documents"] = [
        d for d in manifest.get("documents", [])
        if not ((document_hash and d.get("source_hash") == document_hash) or
                (document_name and d.get("source_name") == document_name and not document_hash))
    ]
    manifest["documents"].append({
        "source_name": document_name or "",
        "source_hash": document_hash or "",
        "chunk_ids": ids,
        "chunk_count": len(normalized),
    })

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return [{"chunk_id": n["chunk_id"], "text": n["text"]} for n in normalized]


def save_index_manifest(records, output_path="data/chroma_manifest.json"):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"documents": list(records)}, f, indent=2, ensure_ascii=False)
    return output_path