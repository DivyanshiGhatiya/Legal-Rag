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


def _embed_texts(texts):
    if not texts:
        return []
    client = ollama.Client(host="http://127.0.0.1:11434")
    response = client.embed(model=EMBED_MODEL, input=list(texts))
    embeddings = response.embeddings if hasattr(response, "embeddings") else response.get("embeddings", [])
    return [list(map(float, e)) for e in embeddings]


def document_already_embedded(document_name, document_hash, collection_name=DEFAULT_COLLECTION_NAME, persist_dir=DEFAULT_PERSIST_DIR):
    manifest_path = os.path.join(persist_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data.get("documents", []) if isinstance(data, dict) else data
        for entry in docs:
            if document_hash and entry.get("source_hash") == document_hash:
                return True
            if document_name and entry.get("source_name") == document_name:
                return True
        return False
    except Exception:
        return False


def embed_chunks(chunks, collection_name=DEFAULT_COLLECTION_NAME, persist_dir=DEFAULT_PERSIST_DIR, document_name=None, document_hash=None):
    normalized = []
    for i, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("content") or ""
            cid = chunk.get("chunk_id") or "chunk_{:03d}".format(i)
        else:
            text = str(chunk)
            cid = "chunk_{:03d}".format(i)
        if text:
            normalized.append({"chunk_id": cid, "text": text})

    if not normalized:
        return []
    if document_already_embedded(document_name, document_hash, collection_name, persist_dir):
        return []

    documents = [n["text"] for n in normalized]
    ids = [n["chunk_id"] for n in normalized]
    prefixed = ["search_document: " + t for t in documents]
    embeddings = _embed_texts(prefixed)

    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
    collection = client.get_or_create_collection(name=collection_name)

    for cid, text, emb in zip(ids, documents, embeddings):
        meta = {"source_name": document_name or "", "source_hash": document_hash or ""}
        collection.add(documents=[text], embeddings=[emb], ids=[cid], metadatas=[meta])

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


def retrieve_chunks(query, collection_name=DEFAULT_COLLECTION_NAME, persist_dir=DEFAULT_PERSIST_DIR, top_k=5, document_filter=None):
    if not os.path.isdir(persist_dir):
        return []
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