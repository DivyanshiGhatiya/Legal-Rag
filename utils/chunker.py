"""
AGGRESSIVE Hierarchy-aware chunking for Legal RAG (v3).
This version prioritizes FINDING HEADINGS over perfect accuracy.
Uses multiple detection strategies to ensure headings are caught.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import re
import json
from dataclasses import dataclass


@dataclass
class HeadingMatch:
    """Structured heading information"""
    start_index: int
    end_index: int
    text: str
    level: int
    heading_type: str
    number: Optional[str] = None


# =============================================================================
# AGGRESSIVE HEADING DETECTION - MULTIPLE STRATEGIES
# =============================================================================

def _detect_headings_aggressive(text: str, debug: bool = False) -> List[HeadingMatch]:
    """
    Detect headings using multiple aggressive strategies.
    Returns deduplicated sorted list of headings.
    """
    headings_dict: Dict[int, HeadingMatch] = {}
    
    # STRATEGY 1: Simple numbered patterns (very permissive)
    numbered_matches = _detect_numbered_headings(text)
    for match in numbered_matches:
        if match.start_index not in headings_dict:
            headings_dict[match.start_index] = match
    
    if debug:
        print(f"Strategy 1 (Numbered): {len(numbered_matches)} headings")
    
    # STRATEGY 2: All-caps headings (very permissive)
    caps_matches = _detect_caps_headings(text)
    for match in caps_matches:
        if match.start_index not in headings_dict:
            headings_dict[match.start_index] = match
    
    if debug:
        print(f"Strategy 2 (Caps): {len(caps_matches)} headings")
    
    # STRATEGY 3: Colon-terminated lines
    colon_matches = _detect_colon_headings(text)
    for match in colon_matches:
        if match.start_index not in headings_dict:
            headings_dict[match.start_index] = match
    
    if debug:
        print(f"Strategy 3 (Colon): {len(colon_matches)} headings")
    
    # STRATEGY 4: Legal keywords
    legal_matches = _detect_legal_keyword_headings(text)
    for match in legal_matches:
        if match.start_index not in headings_dict:
            headings_dict[match.start_index] = match
    
    if debug:
        print(f"Strategy 4 (Legal Keywords): {len(legal_matches)} headings")
    
    # STRATEGY 5: Lettered clauses
    lettered_matches = _detect_lettered_clauses(text)
    for match in lettered_matches:
        if match.start_index not in headings_dict:
            headings_dict[match.start_index] = match
    
    if debug:
        print(f"Strategy 5 (Lettered): {len(lettered_matches)} headings")
    
    # Sort by position
    sorted_headings = sorted(headings_dict.values(), key=lambda x: x.start_index)
    return sorted_headings


def _detect_numbered_headings(text: str) -> List[HeadingMatch]:
    """
    Detect numbered headings: 1., 1.1, 1.1.2, etc.
    VERY PERMISSIVE - will catch almost any line starting with a number.
    """
    matches = []
    
    # Pattern: optional whitespace, number/dots, whitespace, text
    # This is intentionally very broad
    pattern = re.compile(
        r"^[ \t]*(\d+(?:\.\d+)*)\s+([^\n]+)$",
        re.MULTILINE
    )
    
    for match in pattern.finditer(text):
        start = match.start()
        end = match.end()
        number = match.group(1)
        text_part = match.group(2).strip()
        heading_text = match.group(0).strip()
        
        # Filter: skip if text is too short or looks like regular paragraph
        if len(text_part) < 2:
            continue
        
        # Calculate level from number of dots
        level = number.count('.') + 1
        
        matches.append(HeadingMatch(
            start_index=start,
            end_index=end,
            text=heading_text,
            level=level,
            heading_type="NUMBERED",
            number=number,
        ))
    
    return matches


def _detect_caps_headings(text: str) -> List[HeadingMatch]:
    """
    Detect ALL-CAPS headings.
    Will match any line that is all caps and not just numbers.
    """
    matches = []
    
    # Pattern: line of all caps, at least 3 characters, not pure digits
    pattern = re.compile(
        r"^([A-Z][A-Z\s\d\-]{2,})$",
        re.MULTILINE
    )
    
    for match in pattern.finditer(text):
        start = match.start()
        end = match.end()
        heading_text = match.group(0).strip()
        
        # Extra filter: skip pure digit lines or too-short lines
        if heading_text.replace(' ', '').replace('-', '').isdigit():
            continue
        if len(heading_text) < 3:
            continue
        
        # Determine level (use simple heuristic)
        # ARTICLE/PART = level 1, generic caps = level 2
        if any(kw in heading_text for kw in ['ARTICLE', 'PART', 'CHAPTER']):
            level = 1
        else:
            level = 2
        
        matches.append(HeadingMatch(
            start_index=start,
            end_index=end,
            text=heading_text,
            level=level,
            heading_type="CAPS",
        ))
    
    return matches


def _detect_colon_headings(text: str) -> List[HeadingMatch]:
    """
    Detect lines ending with colon (common heading format).
    Example: "Payment Terms:", "Definitions:", "Scope of Work:"
    """
    matches = []
    
    # Pattern: starts with letter, ends with colon, reasonable length
    pattern = re.compile(
        r"^([A-Z][a-zA-Z\s\d\-\&\.]{2,50}:)\s*$",
        re.MULTILINE
    )
    
    for match in pattern.finditer(text):
        start = match.start()
        end = match.end()
        heading_text = match.group(0).strip()
        
        matches.append(HeadingMatch(
            start_index=start,
            end_index=end,
            text=heading_text,
            level=2,
            heading_type="COLON",
        ))
    
    return matches


def _detect_legal_keyword_headings(text: str) -> List[HeadingMatch]:
    """
    Detect lines containing legal keywords.
    Examples: ARTICLE, SECTION, DEFINITION, RECITALS, CONDITIONS, etc.
    """
    matches = []
    
    legal_keywords = [
        r'ARTICLE',
        r'SECTION',
        r'PART',
        r'CLAUSE',
        r'DEFINITION',
        r'RECITAL',
        r'CONDITION',
        r'REPRESENTATION',
        r'WARRANTY',
        r'COVENANT',
        r'TERM',
        r'SCHEDULE',
    ]
    
    # Build pattern: any of these keywords, possibly followed by number/text
    keyword_pattern = '|'.join(legal_keywords)
    pattern = re.compile(
        rf"^[ \t]*({keyword_pattern})[^\n]*$",
        re.MULTILINE | re.IGNORECASE
    )
    
    for match in pattern.finditer(text):
        start = match.start()
        end = match.end()
        heading_text = match.group(0).strip()
        
        # Skip if too long (probably not a heading)
        if len(heading_text) > 150:
            continue
        
        # Determine level based on keyword
        if any(kw in heading_text.upper() for kw in ['ARTICLE', 'PART', 'CHAPTER']):
            level = 1
        elif any(kw in heading_text.upper() for kw in ['SECTION']):
            level = 2
        else:
            level = 3
        
        matches.append(HeadingMatch(
            start_index=start,
            end_index=end,
            text=heading_text,
            level=level,
            heading_type="LEGAL_KEYWORD",
        ))
    
    return matches


def _detect_lettered_clauses(text: str) -> List[HeadingMatch]:
    """
    Detect lettered clauses: (a), (i), (A), etc.
    """
    matches = []
    
    # Pattern: (letter/number) followed by text
    pattern = re.compile(
        r"^\s*(\([a-zA-Z0-9]{1,2}\))\s+([^\n]+)$",
        re.MULTILINE
    )
    
    for match in pattern.finditer(text):
        start = match.start()
        end = match.end()
        letter = match.group(1)
        heading_text = match.group(0).strip()
        
        # Skip very short content
        if len(match.group(2).strip()) < 2:
            continue
        
        matches.append(HeadingMatch(
            start_index=start,
            end_index=end,
            text=heading_text,
            level=4,
            heading_type="LETTERED",
            number=letter,
        ))
    
    return matches


# =============================================================================
# HIERARCHY BUILDING - IMPROVED
# =============================================================================

def _build_hierarchy_path(
    headings: List[HeadingMatch],
    heading_index: int,
) -> List[str]:
    """
    Build complete hierarchy path for a heading.
    Includes all ancestors at same or lower level.
    """
    if heading_index >= len(headings):
        return []
    
    current_heading = headings[heading_index]
    hierarchy = []
    
    # Include all preceding headings with equal or lower level
    for i in range(heading_index):
        prev_heading = headings[i]
        
        # Include if: same level (section in same section) or lower level (parent)
        if prev_heading.level <= current_heading.level:
            hierarchy.append(prev_heading.text)
    
    return hierarchy


# =============================================================================
# SPLIT BY HEADINGS
# =============================================================================

def _split_by_headings(
    text: str,
    headings: List[HeadingMatch],
) -> List[Tuple[str, List[str], Optional[str]]]:
    """
    Split text into blocks delimited by headings.
    Returns: (block_text, hierarchy_path, section_number)
    """
    if not headings:
        return [(text, [], None)]
    
    blocks: List[Tuple[str, List[str], Optional[str]]] = []
    
    for i, heading in enumerate(headings):
        # Block starts at heading, ends at next heading
        start = heading.start_index
        end = headings[i + 1].start_index if i + 1 < len(headings) else len(text)
        
        block_text = text[start:end].strip()
        hierarchy = _build_hierarchy_path(headings, i)
        
        blocks.append((block_text, hierarchy, heading.number))
    
    return blocks


# =============================================================================
# SEMANTIC CHUNKING (Same as before, just re-included)
# =============================================================================

def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Protect abbreviations
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Ltd|Co|Corp|U\.S|U\.K|e\.g|i\.e|etc)\.',
                  lambda m: m.group(1).replace('.', '§'), text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Restore abbreviations
    sentences = [s.replace('§', '.') for s in sentences]
    
    # Clean up
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _semantic_chunk_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into chunks by combining sentences."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not (0 <= overlap < chunk_size):
        raise ValueError("must have 0 <= overlap < chunk_size")
    
    sentences = _split_into_sentences(text)
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence) + 1
        
        if current_chunk and current_length + sentence_length > chunk_size:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Create overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) + 1 <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def _sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Character-based sliding window fallback."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not (0 <= overlap < chunk_size):
        raise ValueError("must have 0 <= overlap < chunk_size")
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start < 0:
            start = 0
    
    return chunks


# =============================================================================
# METADATA EXTRACTION
# =============================================================================

def _extract_legal_metadata(text: str, hierarchy: List[str]) -> Dict[str, Any]:
    """Extract legal-specific metadata from chunk."""
    metadata = {}
    
    # Dates
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4}'
    dates = re.findall(date_pattern, text)
    if dates:
        metadata["dates"] = dates[:3]
    
    # Parties
    party_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|LLC|Ltd|Corp|Company)))\b'
    parties = list(set(re.findall(party_pattern, text)))
    if parties:
        metadata["parties"] = parties[:5]
    
    # Defined terms
    defined_terms = re.findall(r'"([^"]+)"', text[:500])
    if defined_terms:
        metadata["defined_terms"] = defined_terms[:5]
    
    # Definition section
    if hierarchy and any(word in hierarchy[-1].upper() for word in ["DEFINITION", "DEFINED"]):
        metadata["is_definition_section"] = True
    
    # Obligation density
    obligation_count = len(re.findall(r'\b(shall|must|may not|will|required to)\b', text, re.IGNORECASE))
    if obligation_count > 0:
        metadata["obligation_density"] = obligation_count / max(1, len(text.split()))
    
    return metadata


# =============================================================================
# MAIN PUBLIC API
# =============================================================================

def create_chunks_hierarchical(
    doc_meta: Dict[str, Any],
    cleaned_text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    semantic_chunking: bool = True,
    debug: bool = False,  # NEW: Enable debug output
) -> List[Dict[str, Any]]:
    """
    Create hierarchical chunks using AGGRESSIVE heading detection.
    
    Parameters
    ----------
    doc_meta : dict
        Document metadata
    cleaned_text : str
        Cleaned text
    chunk_size : int
        Target chunk size (default 1000)
    overlap : int
        Overlap between chunks (default 200)
    semantic_chunking : bool
        Use semantic chunking (default True)
    debug : bool
        Print debug information (default False)
    
    Returns
    -------
    List[dict]
        List of chunk dictionaries with hierarchy
    """
    if not cleaned_text:
        return []
    
    # AGGRESSIVE heading detection
    headings = _detect_headings_aggressive(cleaned_text, debug=debug)
    
    if debug:
        print(f"\n✓ Total headings detected: {len(headings)}")
        for i, h in enumerate(headings[:10]):
            print(f"  {i+1}. [{h.heading_type:15}] Level {h.level} | {h.text[:60]}")
    
    # Split by headings
    blocks = _split_by_headings(cleaned_text, headings)
    
    # Generate document ID
    file_name = doc_meta.get("file_name", "unknown")
    document_id = os.path.splitext(file_name)[0]
    
    # Create chunks
    all_chunks: List[Dict[str, Any]] = []
    chunk_counter = 1
    
    for block_text, hierarchy, section_number in blocks:
        # Choose chunking strategy
        if len(block_text) <= chunk_size:
            chunks_to_add = [block_text]
        elif semantic_chunking:
            try:
                chunks_to_add = _semantic_chunk_split(block_text, chunk_size, overlap)
            except:
                chunks_to_add = _sliding_window(block_text, chunk_size, overlap)
        else:
            chunks_to_add = _sliding_window(block_text, chunk_size, overlap)
        
        for chunk_text in chunks_to_add:
            chunk_id = f"{document_id}_CHUNK_{chunk_counter:04d}"
            chunk_counter += 1
            
            # Extract section info
            section = ""
            subsection = ""
            if len(hierarchy) >= 2:
                section = hierarchy[-2]
            if len(hierarchy) >= 1:
                subsection = hierarchy[-1]
            
            # Extract metadata
            legal_metadata = _extract_legal_metadata(chunk_text, hierarchy)
            
            chunk_obj = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "file_name": file_name,
                "section": section,
                "subsection": subsection,
                "chunk_index": chunk_counter - 1,
                "chunk_length": len(chunk_text),
                "text": chunk_text,
                "metadata": {
                    **doc_meta,
                    "hierarchy": hierarchy,
                    "section_number": section_number,
                    "hierarchy_depth": len(hierarchy),
                    **legal_metadata,
                },
            }
            all_chunks.append(chunk_obj)
    
    if debug:
        print(f"\n✓ Created {len(all_chunks)} chunks from {len(blocks)} blocks")
    
    return all_chunks


def save_chunked(
    chunks: List[Dict[str, Any]],
    output_dir: str = "chunked_data",
) -> str:
    """Save chunks to JSON."""
    if not chunks:
        raise ValueError("No chunks to save.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    doc_id = chunks[0]["document_id"]
    out_filename = f"{doc_id}_chunks.json"
    out_path = os.path.join(output_dir, out_filename)
    
    payload = {
        "document_id": doc_id,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    return os.path.abspath(out_path)


def validate_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate chunk structure and return metrics."""
    if not chunks:
        return {"valid": False, "error": "No chunks provided"}
    
    metrics = {
        "valid": True,
        "total_chunks": len(chunks),
        "chunks_with_hierarchy": len([c for c in chunks if c["metadata"]["hierarchy"]]),
        "avg_chunk_size": sum(c.get("chunk_length", 0) for c in chunks) / len(chunks),
        "min_chunk_size": min(c.get("chunk_length", 0) for c in chunks),
        "max_chunk_size": max(c.get("chunk_length", 0) for c in chunks),
        "hierarchy_depth_avg": sum(c.get("metadata", {}).get("hierarchy_depth", 0) for c in chunks) / len(chunks),
    }
    
    # Verify required fields
    required_fields = ["chunk_id", "text", "metadata"]
    for chunk in chunks:
        for field in required_fields:
            if field not in chunk:
                metrics["valid"] = False
                metrics["error"] = f"Missing required field: {field}"
                break
    
    return metrics


# =============================================================================
# SIMPLE TEST
# =============================================================================

if __name__ == "__main__":
    test_text = """
SERVICE AGREEMENT

This Service Agreement is entered into as of January 15, 2024, between Acme Corporation and TechVenture LLC.

ARTICLE I - DEFINITIONS

Section 1.1 Defined Terms.

(a) "Company" means Acme Corporation.

(b) "Services" means the services described in Schedule A.

ARTICLE II - SERVICES

Section 2.1 Scope.

The Service Provider shall provide:

(i) Strategic consulting;

(ii) Implementation support.

Section 2.2 Fees.

Client shall pay $150,000.

ARTICLE III - CONFIDENTIALITY

Section 3.1 Obligations.

Each party shall protect the other's Confidential Information.
    """
    
    doc_meta = {"file_name": "test_agreement.pdf", "num_pages": 5}
    
    print("Testing AGGRESSIVE heading detection...\n")
    chunks = create_chunks_hierarchical(doc_meta, test_text, debug=True)
    
    print("\n" + "="*80)
    print("CHUNKS WITH HIERARCHY")
    print("="*80)
    
    for chunk in chunks:
        hierarchy = chunk["metadata"]["hierarchy"]
        breadcrumb = " → ".join(hierarchy) if hierarchy else "[NO HIERARCHY]"
        print(f"\n{chunk['chunk_id']}")
        print(f"  Hierarchy: {breadcrumb}")
        print(f"  Length: {chunk['chunk_length']}")
        print(f"  Text: {chunk['text'][:100]}...")
    
    validation = validate_chunks(chunks)
    print(f"\n✓ Validation: {validation}")