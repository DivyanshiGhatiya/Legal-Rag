import logging
import os
import re
import threading
import time
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from utils.embedding import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
    embed_query,
    get_collection,
)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATE_LIMIT = max(
    1, int(os.getenv("LEGAL_RAG_RERANK_CANDIDATES", "20"))
)

_bm25_cache = {}
_cross_encoder = None
_bm25_lock = threading.Lock()
_cross_encoder_lock = threading.Lock()
logger = logging.getLogger(__name__)


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                started = time.perf_counter()
                try:
                    _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
                finally:
                    logger.info(
                        "pipeline_stage=cross_encoder_load duration_seconds=%.3f",
                        time.perf_counter() - started,
                    )
    return _cross_encoder


def _load_corpus(collection_name=DEFAULT_COLLECTION_NAME, persist_dir=DEFAULT_PERSIST_DIR, force_reload=False):
    cache_key = (collection_name, persist_dir)
    with _bm25_lock:
        if not force_reload and cache_key in _bm25_cache:
            logger.info("pipeline_cache=bm25 status=hit")
            return _bm25_cache[cache_key]

        started = time.perf_counter()
        try:
            collection = get_collection(collection_name, persist_dir)
            raw = collection.get(include=["documents", "metadatas"])

            ids = raw.get("ids", [])
            documents = raw.get("documents", [])
            metadatas = [metadata or {} for metadata in raw.get("metadatas", [])]

            tokenized = [_tokenize(doc) for doc in documents]
            bm25 = BM25Okapi(tokenized) if tokenized else None

            corpus = {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "bm25": bm25,
                "id_to_index": {cid: i for i, cid in enumerate(ids)},
            }
            _bm25_cache[cache_key] = corpus
            return corpus
        finally:
            logger.info(
                "pipeline_cache=bm25 status=%s duration_seconds=%.3f",
                "refresh" if force_reload else "miss",
                time.perf_counter() - started,
            )


def refresh_corpus_cache(collection_name=DEFAULT_COLLECTION_NAME, persist_dir=DEFAULT_PERSIST_DIR):
    """Call this after embedding new documents so BM25 picks up new chunks."""
    return _load_corpus(collection_name, persist_dir, force_reload=True)


def _matches_filters(meta, document_filter=None, section_filter=None):
    if document_filter and meta.get("source_name") != document_filter:
        return False
    if section_filter and section_filter.lower() not in (meta.get("section", "") or "").lower():
        return False
    return True


def dense_search(query, top_k=20, collection_name=DEFAULT_COLLECTION_NAME,
                  persist_dir=DEFAULT_PERSIST_DIR, document_filter=None):
    started = time.perf_counter()
    try:
        return _dense_search(
            query, top_k, collection_name, persist_dir, document_filter
        )
    finally:
        logger.info(
            "pipeline_stage=dense_search duration_seconds=%.3f",
            time.perf_counter() - started,
        )


def _dense_search(query, top_k, collection_name, persist_dir, document_filter):
    collection = get_collection(collection_name, persist_dir)
    query_vec = embed_query(query)

    where = {"source_name": document_filter} if document_filter else None

    query_started = time.perf_counter()
    try:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    finally:
        logger.info(
            "pipeline_stage=chroma_dense_query duration_seconds=%.3f",
            time.perf_counter() - query_started,
        )

    output = []
    if not results or not results.get("ids") or not results["ids"][0]:
        return output

    for i, chunk_id in enumerate(results["ids"][0], start=1):
        result_index = i - 1
        meta = results["metadatas"][0][result_index] if results.get("metadatas") else {}
        meta = meta or {}
        output.append({
            "chunk_id": chunk_id,
            "text": results["documents"][0][result_index] if results.get("documents") else "",
            "dense_score": 1.0 - float(results["distances"][0][result_index]),
            "metadata": meta,
            "dense_rank": i,
            "retrieval_stages": ["DENSE"],
        })
    return output


def bm25_search(query, top_k=20, collection_name=DEFAULT_COLLECTION_NAME,
                 persist_dir=DEFAULT_PERSIST_DIR, document_filter=None, section_filter=None):
    started = time.perf_counter()
    try:
        return _bm25_search(
            query, top_k, collection_name, persist_dir, document_filter, section_filter
        )
    finally:
        logger.info(
            "pipeline_stage=bm25_search duration_seconds=%.3f",
            time.perf_counter() - started,
        )


def _bm25_search(query, top_k, collection_name, persist_dir, document_filter, section_filter):
    corpus = _load_corpus(collection_name, persist_dir)
    if not corpus["bm25"]:
        return []

    tokenized_query = _tokenize(query)
    scores = corpus["bm25"].get_scores(tokenized_query)

    scored = []
    for idx, score in enumerate(scores):
        meta = corpus["metadatas"][idx]
        if not _matches_filters(meta, document_filter, section_filter):
            continue
        scored.append((idx, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:top_k]

    output = []
    for rank, (idx, score) in enumerate(scored, start=1):
        output.append({
            "chunk_id": corpus["ids"][idx],
            "text": corpus["documents"][idx],
            "bm25_score": float(score),
            "metadata": corpus["metadatas"][idx],
            "bm25_rank": rank,
            "retrieval_stages": ["BM25"],
        })
    return output


def rrf_fuse(dense_results, bm25_results, k=60, top_n=20):
    """
    Reciprocal Rank Fusion. Rank-based, so it works even though dense
    similarity and BM25 scores live on completely different scales.
    """
    fused_scores = {}
    chunk_data = {}

    for rank, item in enumerate(dense_results):
        cid = item["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunk_data[cid] = dict(item)

    for rank, item in enumerate(bm25_results):
        cid = item["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if cid not in chunk_data:
            chunk_data[cid] = dict(item)
        else:
            chunk_data[cid]["bm25_score"] = item.get("bm25_score")
            chunk_data[cid]["bm25_rank"] = item.get("bm25_rank")
        chunk_data[cid]["retrieval_stages"] = sorted(
            set(chunk_data[cid].get("retrieval_stages", [])) | {"BM25"}
        )

    ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

    fused = []
    for rank, cid in enumerate(ranked_ids[:top_n], start=1):
        entry = dict(chunk_data[cid])
        entry["rrf_score"] = fused_scores[cid]
        entry["rrf_rank"] = rank
        entry["retrieval_stages"] = sorted(
            set(entry.get("retrieval_stages", [])) | {"RRF"}
        )
        fused.append(entry)
    return fused


def rerank(query, candidates, top_n=6):
    if not candidates:
        return []

    cross_encoder = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    started = time.perf_counter()
    try:
        scores = cross_encoder.predict(pairs)
    finally:
        logger.info(
            "pipeline_stage=cross_encoder_rerank duration_seconds=%.3f candidates=%d",
            time.perf_counter() - started,
            len(pairs),
        )

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)
        c["retrieval_stages"] = sorted(
            set(c.get("retrieval_stages", [])) | {"CROSS_ENCODER"}
        )

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    final = reranked[:top_n]
    for rank, item in enumerate(final, start=1):
        item["final_rank"] = rank
    return final


def retrieve(
    query,
    collection_name=DEFAULT_COLLECTION_NAME,
    persist_dir=DEFAULT_PERSIST_DIR,
    document_filter=None,
    section_filter=None,
    top_k_dense=20,
    top_k_bm25=20,
    top_n_fused=20,
    top_n_final=6,
    use_reranker=True,
):
    """
    Full hybrid retrieval pipeline:
    dense (Chroma) + BM25 -> RRF fusion -> optional metadata filter -> cross-encoder rerank
    """
    started = time.perf_counter()
    try:
        dense_results = dense_search(
            query, top_k=top_k_dense, collection_name=collection_name,
            persist_dir=persist_dir, document_filter=document_filter,
        )
        bm25_results = bm25_search(
            query, top_k=top_k_bm25, collection_name=collection_name,
            persist_dir=persist_dir, document_filter=document_filter, section_filter=section_filter,
        )

        stage_started = time.perf_counter()
        fused = rrf_fuse(
            dense_results,
            bm25_results,
            top_n=min(top_n_fused, RERANK_CANDIDATE_LIMIT),
        )
        logger.info(
            "pipeline_stage=rrf_fusion duration_seconds=%.3f candidates=%d",
            time.perf_counter() - stage_started,
            len(fused),
        )

        if section_filter:
            fused = [f for f in fused if _matches_filters(f.get("metadata", {}), None, section_filter)]

        if use_reranker:
            final = rerank(query, fused, top_n=top_n_final)
        else:
            final = fused[:top_n_final]
            for rank, item in enumerate(final, start=1):
                item["final_rank"] = rank

        output = []
        for item in final:
            meta = item.get("metadata", {})
            output.append({
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "document_name": meta.get("source_name", "?"),
                "file_name": meta.get("source_name", "?"),
                "metadata": meta,
                "section": meta.get("section", ""),
                "subsection": meta.get("subsection", ""),
                "hierarchy": meta.get("hierarchy", ""),
                "dense_score": item.get("dense_score"),
                "bm25_score": item.get("bm25_score"),
                "rrf_score": item.get("rrf_score"),
                "rerank_score": item.get("rerank_score"),
                "dense_rank": item.get("dense_rank"),
                "bm25_rank": item.get("bm25_rank"),
                "rrf_rank": item.get("rrf_rank"),
                "final_rank": item.get("final_rank"),
                "retrieval_stages": item.get("retrieval_stages", []),
            })
        return output
    finally:
        logger.info(
            "pipeline_stage=retrieval_total duration_seconds=%.3f",
            time.perf_counter() - started,
        )


if __name__ == "__main__":
    query = "What are the confidentiality obligations of the parties?"
    results = retrieve(query, top_n_final=5)
    for r in results:
        print(f"\n{r['chunk_id']}  (rerank={r['rerank_score']})")
        print(f"  section: {r['section']} / {r['subsection']}")
        print(f"  {r['text'][:150]}...")