import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

try:
    from langchain_community.vectorstores import FAISS
    import faiss

    FAISS_AVAILABLE = True
except ImportError as e:
    FAISS_AVAILABLE = False
    FAISS_ERROR = str(e)

try:
    import importlib
    # Only mark Chroma available if the chromadb package is installed
    if importlib.util.find_spec("chromadb") is not None:
        CHROMA_AVAILABLE = True
    else:
        CHROMA_AVAILABLE = False
        CHROMA_ERROR = "chromadb not installed"
except Exception as e:
    CHROMA_AVAILABLE = False
    CHROMA_ERROR = str(e)

import os
from dotenv import load_dotenv
import re
import uuid
import numpy as np
import logging

# Optional Hugging Face Inference client (used only if available and enabled)
try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_HUB_AVAILABLE = True
except Exception:
    InferenceClient = None
    HUGGINGFACE_HUB_AVAILABLE = False


# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Setup basic logging
logger = logging.getLogger("pdf_chat_assistant")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def extract_text_from_pdf(uploaded_file):
    """Enhanced PDF text extraction with multiple methods"""
    text = ""

    try:
        # Method 1: Try PyPDF2 first (faster)
        try:
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages):
                content = page.extract_text()
                if content and content.strip():
                    text += f"\n--- Page {page_num + 1} ---\n{content}\n"
        except Exception as e:
            st.warning(f"PyPDF2 extraction failed: {e}. Trying alternative method...")

        # Method 2: If PyPDF2 fails or gives poor results, try pdfplumber
        if len(text.strip()) < 100:  # If we got very little text
            uploaded_file.seek(0)  # Reset file pointer
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    text = ""
                    for page_num, page in enumerate(pdf.pages):
                        content = page.extract_text()
                        if content and content.strip():
                            text += f"\n--- Page {page_num + 1} ---\n{content}\n"
            except Exception as e:
                st.warning(f"pdfplumber extraction also failed: {e}")

        # Clean and normalize the text
        if text.strip():
            text = clean_extracted_text(text)
        else:
            st.error("Could not extract readable text from this PDF. The PDF might be image-based or corrupted.")

    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

    return text


def clean_extracted_text(text):
    # Remove page markers
    text = re.sub(r'--- Page \d+ ---', '', text)

    # Fix line breaks
    text = re.sub(r'\n+', '\n', text)

    # Fix broken words
    text = re.sub(r'-\n', '', text)

    # Space fixes
    text = re.sub(r'\s+', ' ', text)

    # Remove junk symbols
    text = re.sub(r'[•♦■►]', '', text)

    return text.strip()


def assess_text_quality(text):
    """Return a 0-1 score indicating how readable the extracted text is."""
    if not text:
        return 0.0

    sample = text[:5000]
    if not sample:
        return 0.0

    alpha = sum(ch.isalpha() for ch in sample)
    spaces = sum(ch.isspace() for ch in sample)
    total = len(sample)
    weird = sum(not (ch.isalnum() or ch.isspace() or ch in ".,;:!?-()[]'\"") for ch in sample)

    if total == 0:
        return 0.0

    alpha_ratio = alpha / total
    space_ratio = spaces / total
    weird_ratio = weird / total

    score = alpha_ratio * 0.7 + space_ratio * 0.3 - weird_ratio
    return max(0.0, min(1.0, score))
def simple_text_search(chunks, query, max_results=6):
    """Improved fallback text search when vector store is not available"""
    if not chunks:
        return []

    query_words = [word.lower() for word in query.split() if len(word) > 2]
    scored_chunks = []

    for chunk in chunks:
        chunk_text = chunk.lower() if isinstance(chunk, str) else chunk
        score = 0

        # Exact phrase match (highest priority)
        if query.lower() in chunk_text:
            score += 20

        # Individual word matches
        for word in query_words:
            word_count = chunk_text.count(word)
            score += word_count * 3  # Increased weight for word matches

        # Bonus for multiple word matches in same chunk
        word_matches = sum(1 for word in query_words if word in chunk_text)
        if word_matches >= len(query_words) * 0.7:
            score += 10

        # Bonus for structural terms
        structural_terms = ['unit', 'chapter', 'section', 'module', 'lesson', 'topic']
        if any(term in query.lower() for term in structural_terms):
            for term in structural_terms:
                if term in chunk_text:
                    score += 5

        if score > 0:
            # Create a simple chunk-like object
            class SimpleChunk:
                def __init__(self, content):
                    self.page_content = content

            scored_chunks.append((score, SimpleChunk(chunk)))

    # Sort by score and return top chunks
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:max_results]]


def find_most_relevant_chunks(chunks, query, max_chunks=5):
    """Find the most relevant chunks with improved scoring"""
    if not chunks:
        return []

    query_words = [word.lower() for word in query.split() if len(word) > 2]
    scored_chunks = []
    seen_content = set()  # To avoid duplicates

    for chunk in chunks:
        if not hasattr(chunk, 'page_content'):
            continue

        chunk_text = chunk.page_content.lower()

        # Skip very short or duplicate chunks
        if len(chunk.page_content.strip()) < 30:
            continue

        # Simple deduplication check
        chunk_signature = chunk.page_content[:100].lower().strip()
        if chunk_signature in seen_content:
            continue
        seen_content.add(chunk_signature)

        score = 0

        # Exact phrase match gets highest score
        if query.lower() in chunk_text:
            score += 25

        # Count individual word matches
        word_matches = 0
        for word in query_words:
            if word in chunk_text:
                word_count = chunk_text.count(word)
                word_matches += min(word_count, 5)  # Cap to avoid spam
        score += word_matches * 3

        # Bonus for multiple words from query appearing together
        if word_matches >= len(query_words) * 0.6:
            score += 15

        # Bonus for structural keywords matching query
        structural_terms = ['unit', 'chapter', 'section', 'module', 'lesson', 'topic', 'objective']
        query_has_structure = any(term in query.lower() for term in structural_terms)
        if query_has_structure:
            for term in structural_terms:
                if term in chunk_text:
                    score += 10

        # Bonus for content that looks like headings or important content
        lines = chunk.page_content.split('\n')
        for line in lines:
            line_clean = line.strip()
            if line_clean and (line_clean.isupper() or ':' in line_clean):
                score += 2

        if score > 0:
            scored_chunks.append((score, chunk))

    # Sort by score and return top unique chunks
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:max_chunks]]


def generate_comprehensive_answer(query, relevant_chunks, full_context):

    """Generate a comprehensive answer using the most relevant chunks"""
    if not relevant_chunks:
        return "❌ No relevant content found for your query. Please try rephrasing your question or check if the content exists in the PDF."

    # Combine and clean the most relevant content
    combined_content = merge_and_clean_chunks(relevant_chunks)

    if not combined_content or len(combined_content.strip()) < 20:
        return "❌ Could not extract sufficient relevant content. Please try a different question."

    # Apply specific extraction logic based on query type
    query_lower = query.lower().strip()

    # Unit/Chapter/Section queries - improved pattern matching
    unit_match = re.search(r'unit\s*(\d+)', query_lower)
    chapter_match = re.search(r'chapter\s*(\d+)', query_lower)
    section_match = re.search(r'section\s*(\d+)', query_lower)

    if unit_match or chapter_match or section_match:
        number = (unit_match or chapter_match or section_match).group(1)
        structure_type = 'unit' if unit_match else ('chapter' if chapter_match else 'section')
        return extract_structural_content_improved(combined_content, structure_type, number, query)

    # Definition queries
    if any(word in query_lower for word in ['what is', 'define', 'definition', 'explain', 'meaning']):
        return extract_definition_content_improved(combined_content, query)

    # List/enumeration queries
    if any(word in query_lower for word in ['list', 'types', 'kinds', 'examples', 'topics', 'contents']):
        return extract_list_content_improved(combined_content, query)

    # Procedure/how-to queries
    if any(word in query_lower for word in ['how to', 'procedure', 'steps', 'method', 'process']):
        return extract_procedure_content_improved(combined_content, query)

    # Summary/overview queries
    if any(word in query_lower for word in ['summary', 'overview', 'about', 'main points']):
        return extract_summary_content_improved(combined_content, query)

    # Default: comprehensive answer
    return format_comprehensive_response_improved(combined_content, query)

def extract_relevant_snippet(text, query, window=2, max_chars=900):
    """Extract the most relevant sentence snippet from text based on query terms."""
    if not text:
        return ""

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return text[:max_chars]

    query_terms = extract_query_terms(query)
    best_i = 0
    best_score = -1

    for i, sentence in enumerate(sentences):
        s_lower = sentence.lower()
        score = sum(s_lower.count(term) for term in query_terms)
        if score > best_score:
            best_score = score
            best_i = i

    start = max(0, best_i - window)
    end = min(len(sentences), best_i + window + 1)
    snippet = " ".join(sentences[start:end]).strip()

    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "..."

    return snippet


def build_extractive_answer(query, relevant_chunks, max_chunks=3):
    """Return verbatim excerpts from the most relevant chunks for higher accuracy."""
    if not relevant_chunks:
        return "❌ No relevant content found for your query. Please try rephrasing your question or check if the content exists in the PDF."

    def format_answer_text(text):
        # Drop noisy lecture references and empty lines
        lines = [ln.strip() for ln in text.splitlines()]
        cleaned = []
        seen = set()
        for ln in lines:
            if not ln:
                continue
            if re.match(r'(?i)^refer\s+lec', ln):
                continue
            if re.match(r'(?i)^lec[-\s]?\d+', ln):
                continue
            if re.match(r'(?i)^codehelp', ln):
                continue
            if re.match(r'(?i)^what\s+is\s+dbms\s*\??$', ln):
                continue
            if len(ln) < 12:
                continue
            sig = re.sub(r'\s+', ' ', ln).lower()
            if sig in seen:
                continue
            seen.add(sig)
            cleaned.append(ln)

        text = "\n".join(cleaned)

        # Convert numbered lines to bullets
        text = re.sub(r'(?m)^\d+\.\s*', '- ', text)
        text = re.sub(r'(?m)^-\s*-', '- ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def normalize_excerpt_text(text):
        text = re.sub(r'---\s*Page\s*(\d+)\s*---', r'Page \1:', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        # Add missing spaces after punctuation/numbered lists
        text = re.sub(r'(\d)\.(?=[A-Za-z])', r'\1. ', text)
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'(\d+)\.\s*', r'\n\1. ', text)
        return text

    excerpts = []
    for chunk in relevant_chunks[:max_chunks]:
        content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
        snippet = extract_relevant_snippet(content, query)
        if snippet:
            excerpts.append(normalize_excerpt_text(snippet))

    if not excerpts:
        return "❌ Could not extract relevant excerpts from the document."

    response = "\n\n".join(excerpts[:max_chunks])
    return format_answer_text(response)


def extract_query_terms(query):
    stop_words = {
        "what", "why", "how", "explain", "describe", "define", "list", "give",
        "tell", "about", "types", "type", "of", "the", "a", "an", "in", "to",
        "and", "or", "with", "for", "on", "from", "as", "is", "are", "was"
    }

    terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2 and w not in stop_words]
    return terms


def score_chunk_for_query(chunk, terms):
    text = chunk.page_content.lower() if hasattr(chunk, 'page_content') else str(chunk).lower()
    score = sum(1 for t in terms if t in text)

    # Boost for common DB/CS abbreviations
    if "nosql" in terms and ("nosql" in text or "no sql" in text):
        score += 2
    if "acid" in terms and "acid" in text:
        score += 2

    return score


def filter_low_quality_chunks(chunks, query):
    terms = extract_query_terms(query)
    if not terms:
        return chunks

    filtered = []
    for c in chunks:
        score = score_chunk_for_query(c, terms)
        if score >= 1:
            filtered.append(c)

    # Sort by relevance score
    filtered.sort(key=lambda c: score_chunk_for_query(c, terms), reverse=True)
    return filtered


def refine_chunks_by_topic(chunks, query):
    """Narrow chunks for specific topics to avoid cross-topic contamination."""
    q = query.lower()

    # ACID-focused filtering
    if "acid" in q:
        keywords = ["acid", "atomicity", "consistency", "isolation", "durability"]
        narrowed = []
        for c in chunks:
            text = c.page_content.lower() if hasattr(c, 'page_content') else str(c).lower()
            if any(k in text for k in keywords):
                narrowed.append(c)
        if narrowed:
            return narrowed

    # NoSQL-focused filtering
    if "nosql" in q or "no sql" in q:
        keywords = ["nosql", "no sql", "document", "key-value", "wide-column", "graph"]
        narrowed = []
        for c in chunks:
            text = c.page_content.lower() if hasattr(c, 'page_content') else str(c).lower()
            if any(k in text for k in keywords):
                narrowed.append(c)
        if narrowed:
            return narrowed

    return chunks


def validate_answer(answer, chunks):
    hits = 0

    for c in chunks:
        text = c.page_content.lower() if hasattr(c, 'page_content') else str(c).lower()
        if any(
            word in text
            for word in answer.lower().split()
            if len(word) > 4
        ):
            hits += 1

    return hits >= 2


def reject_if_not_found(chunks, query):
    # Reject only if none of the meaningful query terms appear in any chunk.
    terms = extract_query_terms(query)
    if not terms:
        return False

    for c in chunks:
        text = c.page_content.lower() if hasattr(c, 'page_content') else str(c).lower()
        hits = sum(1 for t in terms if t in text)
        if "nosql" in terms and ("nosql" in text or "no sql" in text):
            hits += 1
        if hits >= 1:
            return False

    return True


def build_definition_answer_from_chunks(query, chunks, max_lines=6):
    """Build a cleaner definition-style answer for 'what is' / 'explain' queries."""
    term_patterns = [
        r'what is (.+?)(?:\?|$)',
        r'explain (.+?)(?:\?|$)',
        r'define (.+?)(?:\?|$)'
    ]

    term = None
    q_lower = query.lower().strip()
    for pattern in term_patterns:
        match = re.search(pattern, q_lower)
        if match:
            term = match.group(1).strip()
            break

    if not term:
        return None

    collected = []
    seen = set()
    for c in chunks:
        text = c.page_content if hasattr(c, 'page_content') else str(c)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 25:
                continue
            s_lower = s_clean.lower()
            if term in s_lower and any(k in s_lower for k in [" is ", " means ", " refers", " defined"]):
                sig = re.sub(r'\s+', ' ', s_lower)
                if sig in seen:
                    continue
                seen.add(sig)
                collected.append(s_clean)
            if len(collected) >= max_lines:
                break
        if len(collected) >= max_lines:
            break

    if not collected:
        return None

    return "\n".join(f"- {line}" for line in collected[:max_lines])


def build_simple_summary(text):
    """Create a concise summary from full text using existing heuristics."""
    if not text:
        return "❌ No document text available to summarize."

    try:
        cleaned = re.sub(r'---\s*Page\s*\d+\s*---', ' ', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return extract_summary_content_improved(cleaned, "summary")
    except Exception:
        # Fallback: first few sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        short = " ".join(sentences[:10]).strip()
        return f"## Summary/Overview:\n\n{short}" if short else "❌ No summary could be generated."


def extract_definition_pairs(text, max_pairs=8):
    pairs = []

    if not text:
        return pairs

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    for line in lines:

        if len(line) < 40:
            continue

        if ':' in line and line.count(':') == 1:

            term, definition = line.split(':', 1)

            term = term.strip()
            definition = definition.strip()

            if len(term.split()) > 5:
                continue

            if len(definition.split()) < 6:
                continue

            if any(x in term.lower() for x in ["lec", "unit", "page"]):
                continue

            pairs.append((term, definition))

        if len(pairs) >= max_pairs:
            break

    # Fallback: "is" pattern
    if len(pairs) < max_pairs:

        sentences = re.split(r'[.!?]+', text)

        for s in sentences:

            s = s.strip()

            if len(s) < 30:
                continue

            if " is " in s.lower():

                parts = s.split(" is ", 1)

                term = parts[0].strip()
                definition = parts[1].strip()

                if len(term.split()) <= 4 and len(definition.split()) >= 6:
                    pairs.append((term, definition))

            if len(pairs) >= max_pairs:
                break

    return pairs


def build_quiz_from_pairs(pairs, max_q=5):
    """Build short-answer quiz questions from definition pairs."""
    if not pairs:
        return "❌ Could not derive quiz questions from the document."

    quiz = "### Quiz (Short Answer)\n\n"
    for i, (term, definition) in enumerate(pairs[:max_q], 1):
        if definition.lower().startswith(term.lower()):
            definition = definition[len(term):].strip(' :.-')
        quiz += f"**Q{i}.** Define {term}.\n\n"
        quiz += f"**Answer:** {definition}\n\n"
    return quiz


def build_quiz_from_chunks(chunks, max_q=5):
    """Generate a simple quiz by extracting clean, informative sentences."""
    if not chunks:
        return "❌ No relevant content found to build a quiz."

    text = " ".join([c.page_content if hasattr(c, 'page_content') else str(c) for c in chunks])
    sentences = re.split(r'(?<=[.!?])\s+', text)
    questions = []
    seen = set()

    for s in sentences:
        s = re.sub(r'\s+', ' ', s).strip()
        if len(s) < 45 or len(s) > 180:
            continue
        if '?' in s:
            continue
        if s.lower().startswith("unit ") or s.lower().startswith("chapter "):
            continue
        sig = s[:80].lower()
        if sig in seen:
            continue
        seen.add(sig)
        questions.append(s)
        if len(questions) >= max_q:
            break

    if not questions:
        return "❌ Could not generate quiz questions from the document text."

    quiz = "### Quiz (Short Answer)\n\n"
    for i, s in enumerate(questions, 1):
        quiz += f"**Q{i}.** Explain: {s}\n\n"
        quiz += "**Answer:** (Write your response in your own words.)\n\n"

    return quiz


def build_flashcards_from_pairs(pairs, max_cards=6):
    """Build flashcards from definition pairs."""
    if not pairs:
        return "❌ Could not derive flashcards from the document."

    cards = "### Flashcards\n\n"
    for i, (term, definition) in enumerate(pairs[:max_cards], 1):
        cards += f"**Card {i}:** {term}\n\n"
        cards += f"**Back:** {definition}\n\n"
    return cards

def merge_and_clean_chunks(chunks):
    """Merge chunks intelligently and remove duplicates"""
    if not chunks:
        return ""

    # Collect all content with better deduplication
    all_content = []
    seen_sentences = set()

    for chunk in chunks:
        if not hasattr(chunk, 'page_content'):
            continue

        # Split into sentences for better deduplication
        sentences = re.split(r'[.!?]+', chunk.page_content)

        for sentence in sentences:
            sentence_clean = sentence.strip()
            if (sentence_clean and
                    len(sentence_clean) > 15 and
                    sentence_clean not in seen_sentences and
                    not sentence_clean.startswith('---')):  # Skip page separators

                seen_sentences.add(sentence_clean)
                all_content.append(sentence_clean)

    return '. '.join(all_content)


def extract_structural_content_improved(content, structure_type, number, query):
    """Extract content for specific units/chapters/sections with better parsing"""
    if not content:
        return f"❌ No content found for {structure_type} {number}"

    # Split content into sentences for better processing
    sentences = re.split(r'[.!?]+', content)
    relevant_content = []

    # Look for the specific structure with multiple patterns
    target_patterns = [
        f"{structure_type} {number}",
        f"{structure_type}-{number}",
        f"{structure_type} {number}:",
        f"{structure_type}.{number}",
        f"{number}.",
        f"{number})"
    ]

    # Find content related to the target structure
    found_target = False

    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean or len(sentence_clean) < 10:
            continue

        sentence_lower = sentence_clean.lower()

        # Check if this sentence mentions our target
        if any(pattern.lower() in sentence_lower for pattern in target_patterns):
            found_target = True
            relevant_content.append(f"**{sentence_clean}**")
            continue

        # If we found the target, continue collecting related content
        if found_target and len(relevant_content) < 15:
            # Stop if we hit the next unit/chapter
            next_num = str(int(number) + 1) if number.isdigit() else None
            if next_num:
                stop_patterns = [f"{structure_type} {next_num}", f"{structure_type}-{next_num}"]
                if any(pattern.lower() in sentence_lower for pattern in stop_patterns):
                    break

            relevant_content.append(sentence_clean)

    if relevant_content:
        result = f"## {structure_type.title()} {number} Content:\n\n"
        result += "\n\n".join(relevant_content[:12])  # Limit to prevent overwhelming
        return result
    else:
        # Fallback: look for any content mentioning the number
        fallback_content = []
        for sentence in sentences[:20]:  # Check first 20 sentences
            if number in sentence and len(sentence.strip()) > 15:
                fallback_content.append(sentence.strip())

        if fallback_content:
            result = f"## Related to {structure_type.title()} {number}:\n\n"
            result += "\n\n".join(fallback_content[:5])
            return result
        else:
            return f"❌ {structure_type.title()} {number} not found in the document. The content may use different formatting or numbering."


def extract_definition_content_improved(content, query):
    term_patterns = [
        r'what is (.+?)(?:\?|$)',
        r'define (.+?)(?:\?|$)',
        r'meaning of (.+?)(?:\?|$)',
        r'explain (.+?)(?:\?|$)'
    ]

    term = None
    for pattern in term_patterns:
        match = re.search(pattern, query.lower())
        if match:
            term = match.group(1).strip()
            break

    sentences = re.split(r'[.!?]+', content)

    definitions = []

    if term:
        for s in sentences:
            s = s.strip()

            if len(s) < 20:
                continue

            s_lower = s.lower()

            if term in s_lower and any(
                x in s_lower for x in [" is ", " means ", " defined", " refers"]
            ):
                definitions.append(s)

    if definitions:
        result = "## Definition:\n\n"
        for d in definitions[:4]:
            result += f"• {d}\n\n"
        return result

    # Fallback
    important = [s.strip() for s in sentences if len(s.strip()) > 25]

    result = "## Related Information:\n\n"
    for s in important[:5]:
        result += f"• {s}\n\n"

    return result

def extract_list_content_improved(content, query):
    """Extract lists with better formatting and organization"""
    sentences = re.split(r'[.!?]+', content)
    list_items = []
    numbered_items = []

    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean or len(sentence_clean) < 10:
            continue

        # Detect different types of lists
        if re.match(r'^\d+\.?\s+', sentence_clean):  # Numbered lists
            item = re.sub(r'^\d+\.?\s*', '', sentence_clean)
            numbered_items.append(item)
        elif any(char in sentence_clean for char in ['•', '-', '*']) and len(sentence_clean) < 150:
            list_items.append(sentence_clean)
        elif ':' in sentence_clean and len(sentence_clean) < 100:  # Topic-like items
            list_items.append(sentence_clean)

    result = "## Content List:\n\n"

    if numbered_items:
        result += "**📋 Main Items:**\n"
        for i, item in enumerate(numbered_items[:10], 1):
            result += f"{i}. {item}\n"
        result += "\n"

    if list_items:
        result += "**• Key Points:**\n"
        for item in list_items[:8]:
            result += f"• {item}\n"
        result += "\n"

    if not numbered_items and not list_items:
        # Provide structured content from sentences
        meaningful_sentences = [s.strip() for s in sentences[:10] if len(s.strip()) > 20]
        result += "**📌 Available Content:**\n"
        for i, sentence in enumerate(meaningful_sentences[:8], 1):
            result += f"{i}. {sentence}\n"

    return result


def extract_procedure_content_improved(content, query):
    """Extract procedures with better step identification"""
    sentences = re.split(r'[.!?]+', content)
    steps = []
    procedures = []

    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean:
            continue

        # Look for step-like content
        if (re.match(r'^\d+\.?\s+', sentence_clean) or
                'step' in sentence_clean.lower() or
                'procedure' in sentence_clean.lower() or
                'method' in sentence_clean.lower()):

            if re.match(r'^\d+\.?\s+', sentence_clean):
                steps.append(sentence_clean)
            else:
                procedures.append(sentence_clean)

    result = "## Procedure/Method:\n\n"

    if steps:
        result += "**📋 Steps:**\n"
        for step in steps[:12]:
            result += f"{step}\n"
        result += "\n"

    if procedures:
        result += "**🔧 Methods/Procedures:**\n"
        for proc in procedures[:6]:
            result += f"• {proc}\n"

    if not steps and not procedures:
        # Look for process-related content
        meaningful_sentences = [s.strip() for s in sentences[:8] if len(s.strip()) > 20]
        result += "**Process Information:**\n"
        for sentence in meaningful_sentences[:6]:
            result += f"• {sentence}\n"

    return result


def extract_summary_content_improved(content, query):
    sentences = re.split(r'(?<=[.!?])\s+', content)

    selected = []
    seen = set()

    def normalize_sentence(s):
        def fix_broken_words(text):
            stop_short = {"a", "an", "the", "is", "in", "of", "to", "as", "by", "or", "at", "be", "we", "us", "it", "on", "if", "do", "go", "no", "so", "he", "she", "am", "are", "was"}
            tokens = text.split()
            fixed = []
            i = 0
            while i < len(tokens):
                t = tokens[i]
                if i + 1 < len(tokens):
                    n = tokens[i + 1]
                    if t.isalpha() and n.isalpha():
                        if len(n) <= 2 and n.lower() not in stop_short:
                            t = t + n
                            i += 1
                        elif len(t) <= 2 and t.lower() not in stop_short:
                            t = t + n
                            i += 1
                fixed.append(t)
                i += 1
            return " ".join(fixed)

        s = re.sub(r'\s+', ' ', s).strip()
        s = re.sub(r'\[[^\]]+\]', '', s)  # Remove bracketed citations
        s = re.sub(r'\([^)]*\)', '', s)  # Remove parenthetical noise
        s = re.sub(r'\s*-\s*', ' ', s)  # Remove dash fragments
        s = re.sub(r'\b(i{1,3}|iv|v)\.\b', '', s, flags=re.IGNORECASE)
        s = fix_broken_words(s)
        s = s.strip(' .:-')
        if not s.endswith('.'):
            s = s + '.'
        return s

    for s in sentences:
        s = s.strip()
        if len(s) < 35:
            continue
        if s.lower().startswith("unit ") or s.lower().startswith("chapter "):
            continue
        if s.lower().startswith("page "):
            continue
        if s.count(',') > 6:
            continue
        s = normalize_sentence(s)
        if len(s) < 35:
            continue
        sig = s[:80].lower()
        if sig in seen:
            continue
        seen.add(sig)
        selected.append(s)

    if not selected:
        selected = [normalize_sentence(s) for s in sentences if len(s.strip()) > 30]

    result = "## Summary/Overview:\n\n"
    for s in selected[:12]:
        result += f"• {s}\n\n"

    return result


def handle_generic_overview_query(query, full_text):
    """Provide stable, relevant answers for generic overview questions."""
    if not full_text:
        return None

    q = query.lower().strip()
    summary_patterns = [
        "summary", "overview", "summarize", "give the summary", "brief", "short summary"
    ]
    topic_patterns = [
        "main topics", "topics covered", "main contents", "contents", "what is this document about",
        "objectives"
    ]

    if any(p in q for p in summary_patterns):
        return extract_summary_content_improved(full_text, query)

    if any(p in q for p in topic_patterns):
        topics = extract_available_topics(full_text)
        if topics and topics[0] not in ["Document content not available", "No specific topics detected"]:
            result = "## Main Topics\n\n"
            for t in topics[:10]:
                result += f"• {t}\n"
            return result

        # Fallback to a longer summary if topics are not detectable
        return extract_summary_content_improved(full_text, query)

    return None


def simplify_answer(text):
    prompt = f"""
    Rewrite this in simple student-friendly language.
    Use short sentences.
    Avoid complex words.
    Make it easy to remember.
    Do not add any new information. Only rephrase the given text.

    Text:
    {text}
    """

    if HUGGINGFACE_HUB_AVAILABLE and HF_TOKEN and InferenceClient:
        try:
            client = InferenceClient(api_key=HF_TOKEN)
            return client.text_generation(
                model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                inputs=prompt,
                max_new_tokens=300,
                temperature=0.2
            )
        except Exception:
            return text

    return text


def generate_general_answer(query, model="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_tokens=500):
    """Generate a general answer when the PDF lacks relevant content."""
    if not HUGGINGFACE_HUB_AVAILABLE or not InferenceClient or not HF_TOKEN:
        return None

    prompt = f"""
You are a helpful tutor. Answer the question clearly and concisely.
If you are unsure, say you are not sure.

Question:
{query}
"""

    try:
        client = InferenceClient(api_key=HF_TOKEN)
        response = client.text_generation(
            model=model,
            inputs=prompt,
            max_new_tokens=max_tokens,
            temperature=0.2
        )
        return response.strip()
    except Exception:
        return None


def format_comprehensive_response_improved(content, query):

    return f"""
## 📘 Answer: {query}

### ✅ Simple Explanation
{content}

### 📌 Key Points
• Read carefully  
• Focus on concepts  
• Use examples  

### 📝 Exam Tip
Revise this topic twice before exams.
"""





def generate_query_suggestions(query, full_text):
    """Generate helpful query suggestions based on document content"""
    suggestions = []

    # Check if full_text is valid
    if not full_text or not isinstance(full_text, str):
        return [
            "What is this document about?",
            "What are the main topics covered?",
            "List the contents"
        ]

    # Extract common patterns from the document
    lines = full_text.split('\n')

    # Look for units/chapters with better pattern matching
    units = set()
    chapters = set()

    for line in lines[:100]:  # Check first 100 lines
        line_lower = line.lower().strip()

        # Unit patterns
        unit_matches = re.findall(r'unit\s*(\d+)', line_lower)
        for match in unit_matches:
            if len(units) < 3:
                units.add(f"What are the topics in Unit {match}?")

        # Chapter patterns
        chapter_matches = re.findall(r'chapter\s*(\d+)', line_lower)
        for match in chapter_matches:
            if len(chapters) < 3:
                chapters.add(f"What does Chapter {match} cover?")

    # Add found suggestions
    suggestions.extend(list(units)[:2])
    suggestions.extend(list(chapters)[:2])

    # General suggestions based on content
    content_lower = full_text.lower()

    if 'objective' in content_lower:
        suggestions.append("What are the objectives?")
    if 'assessment' in content_lower:
        suggestions.append("What are the assessment methods?")
    if 'reference' in content_lower:
        suggestions.append("What are the references?")

    # Default suggestions if we don't have specific ones
    default_suggestions = [
        "What is this document about?",
        "What are the main topics covered?",
        "List the main contents",
        "What are the key points?"
    ]

    # Fill up to 5 suggestions
    while len(suggestions) < 5:
        for default in default_suggestions:
            if default not in suggestions:
                suggestions.append(default)
                break
        else:
            break

    return suggestions[:5]


def extract_available_topics(full_text):
    """Extract available topics from the document with better parsing"""
    topics = []

    if not full_text or not isinstance(full_text, str):
        return ["Document content not available"]

    lines = full_text.split('\n')
    seen_topics = set()

    for line in lines[:80]:  # Check first 80 lines for topics
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 10:
            continue

        # Look for topic-like patterns
        patterns_found = []

        # Pattern 1: Numbered items (1. Topic, 2. Topic)
        if re.match(r'^\d+\.?\s+', line_clean):
            topic = re.sub(r'^\d+\.?\s*', '', line_clean)
            patterns_found.append(topic)

        # Pattern 2: Unit/Chapter titles
        elif re.search(r'unit\s*\d+|chapter\s*\d+', line_clean.lower()):
            patterns_found.append(line_clean)

        # Pattern 3: Title: Content format
        elif ':' in line_clean and len(line_clean.split(':')[0]) < 50:
            topic = line_clean.split(':')[0].strip()
            patterns_found.append(topic)

        # Pattern 4: All caps titles (but not too long)
        elif line_clean.isupper() and len(line_clean) < 80:
            patterns_found.append(line_clean.title())

        # Add unique topics
        for topic in patterns_found:
            topic_clean = re.sub(r'\s+', ' ', topic).strip()
            topic_clean = re.sub(r'[•*-]\s*', '', topic_clean).strip()
            topic_clean = topic_clean.strip(':-')

            # Filter noisy fragments
            if any(bad in topic_clean.lower() for bad in ["lec", "page", "unit", "chapter", "come to some conclusion"]):
                continue
            if topic_clean.lower().startswith("e ") or topic_clean.lower().startswith("e."):
                continue
            if topic_clean.endswith("that"):
                continue

            # Normalize case for short titles
            if len(topic_clean) <= 60:
                topic_clean = topic_clean[0].upper() + topic_clean[1:]

            if (len(topic_clean) > 5 and
                    len(topic_clean) < 100 and
                    topic_clean not in seen_topics and
                    len(topic_clean.split()) <= 10):
                seen_topics.add(topic_clean)
                topics.append(topic_clean)

    return topics[:8] if topics else ["No specific topics detected"]


def rerank_chunks_with_embeddings(chunks, query, top_k=6, lambda_param=0.7, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Re-rank retrieved chunks using embeddings + MMR for better diversity and relevance.
    Accepts a list of chunk-like objects with attribute `page_content` and returns top_k chunks.
    """
    if not chunks:
        return []

    texts = [c.page_content if hasattr(c, 'page_content') else str(c) for c in chunks]

    try:
        embedder = HuggingFaceEmbeddings(model_name=model_name)
        if hasattr(embedder, 'embed_documents'):
            emb_docs = embedder.embed_documents(texts)
            query_emb = embedder.embed_query(query)
        else:
            emb_docs = [embedder.embed_query(t) for t in texts]
            query_emb = embedder.embed_query(query)
    except Exception:
        # If embeddings fail, fallback to original scoring
        return find_most_relevant_chunks(chunks, query, max_chunks=top_k)

    emb_matrix = np.array(emb_docs)
    query_vec = np.array(query_emb)

    # cosine similarity helper
    def cos_sim(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # score to document
    sims = [cos_sim(doc, query_vec) for doc in emb_matrix]

    selected_idx = []
    for _ in range(min(top_k, len(texts))):
        best_score = -1e9
        best_i = None
        for i in range(len(texts)):
            if i in selected_idx:
                continue
            sim_to_query = sims[i]
            if not selected_idx:
                mmr_score = sim_to_query
            else:
                max_sim_to_selected = max(cos_sim(emb_matrix[i], emb_matrix[j]) for j in selected_idx)
                mmr_score = lambda_param * sim_to_query - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_i = i

        if best_i is None:
            break
        selected_idx.append(best_i)

    # preserve original order of selected
    selected_idx.sort()
    selected = [chunks[i] for i in selected_idx]
    return selected


def compute_reranker_scores(chunks, query, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Compute raw cosine similarity scores between chunks and query using embeddings.
    Returns list of tuples (chunk, score) in descending score order.
    """
    if not chunks:
        return []

    texts = [c.page_content if hasattr(c, 'page_content') else str(c) for c in chunks]
    try:
        embedder = HuggingFaceEmbeddings(model_name=model_name)
        # prefer document/query API when available
        if hasattr(embedder, 'embed_documents'):
            emb_docs = embedder.embed_documents(texts)
            query_emb = embedder.embed_query(query)
        else:
            emb_docs = [embedder.embed_query(t) for t in texts]
            query_emb = embedder.embed_query(query)
    except Exception as e:
        logger.exception("Embeddings failed in compute_reranker_scores")
        return []

    emb_matrix = np.array(emb_docs)
    query_vec = np.array(query_emb)

    def cos_sim(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    sims = [cos_sim(doc, query_vec) for doc in emb_matrix]
    scored = list(zip(chunks, sims))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def generate_answer_with_hf_llm(query, chunks, model="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_tokens=700):
    """
    Improved RAG Prompt for higher accuracy.
    Uses a more capable 'Instruct' model by default.
    """
    if not HUGGINGFACE_HUB_AVAILABLE or not InferenceClient or not HF_TOKEN:
        return None
    try:
        # Build a robust context block
        context_text = ""
        for i, c in enumerate(chunks):
            content = c.page_content if hasattr(c, 'page_content') else str(c)
            context_text += f"[Source {i+1}]: {content}\n\n"

        # Refined Prompt Engineering
        prompt = f"""<|system|>
You are an expert academic assistant. Use the provided Context to answer the User Question. 
Guidelines:
1. If the answer isn't in the context, say "I cannot find this in the document."
2. Use professional, clear language.
3. Refer to sources as [Source X].
4. Do not make up facts.
<|end|>
<|user|>
Context:
{context_text}

Question: {query}
<|end|>
<|assistant|>"""

        client = InferenceClient(api_key=HF_TOKEN)

        # Using text_generation with specific stop sequences for higher precision
        response = client.text_generation(
            model=model,
            inputs=prompt,
            max_new_tokens=max_tokens,
            temperature=0.1,
            stop_sequences=["<|end|>", "User:"]
        )

        return response.strip()
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return None


# Streamlit app configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📘",
    layout="wide"
)

# Custom UI styling
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px 800px at 10% -10%, rgba(99,102,241,0.18), transparent 60%),
                    radial-gradient(900px 600px at 110% 0%, rgba(14,165,233,0.12), transparent 55%);
    }
    .hero {
        background: linear-gradient(135deg, rgba(30,41,59,0.85), rgba(15,23,42,0.92));
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 18px;
        padding: 22px 26px;
        margin: 8px 0 18px 0;
        box-shadow: 0 10px 30px rgba(2,6,23,0.35);
    }
    .hero-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .hero-sub {
        color: rgba(226,232,240,0.85);
        font-size: 15px;
    }
    .card {
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.25);
    }
    .question-bubble {
        display: inline-block;
        max-width: 90%;
        padding: 10px 16px;
        margin: 6px 0 12px 0;
        border-radius: 999px;
        background: rgba(30,41,59,0.8);
        border: 1px solid rgba(148,163,184,0.2);
        font-weight: 600;
    }
    .answer-wrap {
        border-left: 3px solid rgba(56,189,248,0.7);
        padding-left: 14px;
        margin-top: 6px;
    }
    .section-title {
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 10px 16px;
    }
    .stButton > button {
        border-radius: 10px;
        padding: 8px 14px;
        border: 1px solid rgba(148,163,184,0.25);
    }
    .stFileUploader, .stTextInput, .stNumberInput, .stTextArea {
        background: rgba(2,6,23,0.15);
        border-radius: 12px;
    }
    .question-card {
        margin-top: 10px;
        margin-bottom: 12px;
        border: 1px solid rgba(148,163,184,0.22);
        background: rgba(15,23,42,0.65);
        border-radius: 14px;
        padding: 14px 16px;
    }
    .search-label {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">📘 Universal PDF Chat — Ask Your PDF Anything</div>
        <div class="hero-sub">Accurate answers, smart summaries, quizzes, and flashcards powered by your document.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type="pdf",
        help="Supports all types of PDF files - scanned, text-based, academic papers, reports, etc."
    )

    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB"
        }
        st.json(file_details)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'full_text' not in st.session_state:
    st.session_state.full_text = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'text_quality' not in st.session_state:
    st.session_state.text_quality = None

# Process uploaded file
if uploaded_file is not None:
    # Add a summary button
    st.markdown("---")
    # Check if we need to reprocess the file
    if st.session_state.processed_file != uploaded_file.name:
        st.session_state.processing_complete = False

        with st.spinner("🔍 Processing PDF... This may take a moment for large files."):
            try:
                # Extract text from PDF
                raw_text = extract_text_from_pdf(uploaded_file)

                if not raw_text or len(raw_text.strip()) < 50:
                    st.error(
                        "❌ Could not extract sufficient text from the PDF. Please ensure the PDF contains readable text.")
                    st.stop()

                # Store full text for fallback searches
                st.session_state.full_text = raw_text

                # Assess text quality (detect OCR/garbled extraction)
                st.session_state.text_quality = assess_text_quality(raw_text)
                if st.session_state.text_quality < 0.55:
                    st.warning(
                        "⚠️ The extracted text looks low quality (likely scanned/OCR). "
                        "Answers may be inaccurate. For best results, use a text-based PDF or OCR the file first."
                    )
                # Display text extraction success
                st.success(f"✅ Successfully extracted {len(raw_text)} characters from PDF")

                # Split into chunks with optimized parameters
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=900,
                    chunk_overlap=250,
                    separators=["\n\n", "\n", ". ", "? ", "! "]
                )
                chunks = splitter.split_text(raw_text)
                st.session_state.chunks = chunks

                if not chunks:
                    st.error("❌ Could not create text chunks from the PDF.")
                    st.stop()

                # Create embeddings and vector store
                with st.spinner("🧠 Creating knowledge base..."):
                    vector_store_created = False

                    # Try FAISS first (if available)
                    if FAISS_AVAILABLE and not vector_store_created:
                        try:
                            embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-mpnet-base-v2"
                            )
                            st.session_state.vector_store = FAISS.from_texts(
                                chunks,
                                embedding=embeddings
                            )
                            st.success("✅ Using FAISS vector store")
                            vector_store_created = True
                        except Exception as faiss_error:
                            st.warning(f"FAISS failed: {faiss_error}")

                    # Try Chroma (if available and FAISS failed)
                    if CHROMA_AVAILABLE and not vector_store_created:
                        try:
                            embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-mpnet-base-v2"
                            )
                            collection_name = f"pdf_{uuid.uuid4().hex[:8]}"
                            st.session_state.vector_store = Chroma.from_texts(
                                chunks,
                                embedding=embeddings,
                                collection_name=collection_name
                            )
                            st.success("✅ Using Chroma vector store")
                            vector_store_created = True
                        except Exception as chroma_error:
                            st.warning(f"Chroma failed: {chroma_error}")

                    # Fallback to simple text search
                    if not vector_store_created:
                        st.session_state.vector_store = None
                        st.session_state.chunks = chunks
                        st.info("✅ Using simple text search (no vector embeddings)")

                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.processing_complete = True

                st.success(f"✅ PDF processed successfully! Created {len(chunks)} knowledge chunks.")

            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.write("**Troubleshooting tips:**")
                st.write("- Ensure the PDF is not password-protected")
                st.write("- Try a different PDF file")
                st.write("- Check if the PDF contains readable text (not just images)")
                st.stop()

    # Show interface after upload; disable inputs until processing is complete
    if uploaded_file is not None:
        # Question interface
        st.header("💬 Ask Questions About Your PDF")

        if not st.session_state.processing_complete:
            st.info("Processing your PDF. The question box will enable when ready.")
        # Provide example questions based on document content
        if st.session_state.full_text:
            col1, col2 = st.columns([1, 1])

            with col1:
                with st.expander("💡 Smart Question Suggestions", expanded=True):
                    try:
                        smart_suggestions = generate_query_suggestions("", st.session_state.full_text)
                        st.write("**❓ Try these questions:**")

                        for i, suggestion in enumerate(smart_suggestions):
                            # Use unique keys for buttons to avoid conflicts
                            if st.button(suggestion, key=f"suggestion_{i}_{hash(suggestion)}",
                                         use_container_width=True):
                                st.session_state.current_query = suggestion
                                st.rerun()

                    except Exception as suggestion_error:
                        st.write("Could not generate smart suggestions")

            with col2:
                with st.expander("📋 Available Topics", expanded=True):
                    try:
                        available_topics = extract_available_topics(st.session_state.full_text)
                        st.write("**📌 Ask about these topics:**")

                        for i, topic in enumerate(available_topics[:6]):
                            if topic != "Document content not available" and topic != "No specific topics detected":
                                # Create a question from the topic
                                topic_question = f"What is {topic}?" if not topic.endswith('?') else topic
                                if st.button(f"{topic[:45]}...", key=f"topic_{i}_{hash(topic)}",
                                             use_container_width=True):
                                    st.session_state.current_query = topic_question
                                    st.rerun()

                        if available_topics[0] in ["Document content not available", "No specific topics detected"]:
                            st.write("No specific topics detected in the document")

                    except Exception as topic_error:
                        st.write("Could not extract topics")

        # Initialize current query in session state
        if 'current_query' not in st.session_state:
            st.session_state.current_query = ""

        answer_mode = st.selectbox(
            "Answer style",
            ["Comprehensive (merged summary)", "Extractive (verbatim snippets)"],
            index=0
        )
        # LLM options in UI: allow user to enable/disable remote LLM generation
        use_llm = st.checkbox("Use LLM-based answer generation (Hugging Face Inference)", value=False)
        llm_model = st.text_input("LLM model id (Hugging Face)", value="HuggingFaceTB/SmolLM3-3B", help="Model repo id to use on Hugging Face Inference API")
        # Debugging / tuning controls
        show_reranker_scores = st.checkbox("Show raw reranker scores (debug)", value=False)
        max_context_chars = st.number_input("Max context chars to send to LLM", min_value=500, max_value=60000, value=4000, step=100)

        st.markdown(
            """
            <div class="question-card">
                <div class="search-label">Ask any question about your PDF</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Text input for questions
        user_query = st.text_input(
            "Search",
            value=st.session_state.current_query,
            placeholder="Type your question here...",
            help="Ask anything about the content of your PDF document",
            key="query_input",
            disabled=not st.session_state.processing_complete
        )

        # Clear the session state query after displaying it
        if st.session_state.current_query and user_query == st.session_state.current_query:
            st.session_state.current_query = ""

        # Process the query
        if user_query and user_query.strip():
            if not st.session_state.processing_complete:
                st.warning("Please wait until PDF processing finishes.")
                st.stop()
            if st.session_state.text_quality is not None and st.session_state.text_quality < 0.45:
                st.error(
                    "The extracted text is too garbled to answer reliably. "
                    "Please upload a text-based PDF or run OCR on the document."
                )
                st.stop()
            if st.session_state.vector_store or st.session_state.chunks:
                with st.spinner("🧠 Analyzing document for the most accurate answer..."):
                    try:
                        # Handle generic overview questions with stable answers
                        overview_resp = handle_generic_overview_query(user_query, st.session_state.full_text)
                        if overview_resp:
                            st.subheader("📝 Answer:")
                            st.markdown(overview_resp)
                            st.stop()

                        # 1. Retrieval: Hybrid search (vector + keyword)
                        if st.session_state.vector_store:
                            vector_results = st.session_state.vector_store.similarity_search(
                                user_query,
                                k=15
                            )

                            keyword_results = simple_text_search(
                                st.session_state.chunks,
                                user_query,
                                max_results=10
                            )

                            matching_chunks = list({id(c): c for c in vector_results + keyword_results}.values())

                            relevant_chunks = rerank_chunks_with_embeddings(
                                matching_chunks,
                                user_query,
                                top_k=6,
                                lambda_param=0.7
                            )
                        else:
                            relevant_chunks = simple_text_search(st.session_state.chunks, user_query, max_results=10)

                        # Remove low-quality chunks
                        relevant_chunks = filter_low_quality_chunks(relevant_chunks, user_query)

                        # Topic-aware refinement to avoid cross-topic leakage
                        relevant_chunks = refine_chunks_by_topic(relevant_chunks, user_query)

                        # 2. Generation: Extract -> LLM -> Validate
                        definition_answer = build_definition_answer_from_chunks(user_query, relevant_chunks)
                        base_answer = definition_answer or build_extractive_answer(
                            user_query,
                            relevant_chunks,
                            max_chunks=8
                        )

                        response = base_answer
                        if use_llm:
                            # Only rewrite the extracted answer to avoid hallucinations
                            rewritten = simplify_answer(base_answer)
                            if rewritten:
                                response = rewritten

                        # Auto-reject unknown questions (fallback to AI if enabled)
                        if reject_if_not_found(relevant_chunks, user_query):
                            if use_llm:
                                fallback = generate_general_answer(user_query, model=llm_model, max_tokens=400)
                                response = fallback or "❌ This information is not clearly present in the PDF."
                            else:
                                response = "❌ This information is not clearly present in the PDF."

                        # Confidence validation: fallback to extracted answer if weak
                        is_valid = validate_answer(response, relevant_chunks)
                        if not is_valid:
                            response = base_answer

                        # Display answer
                        st.markdown(
                            f"""
                            <div class="question-bubble">{user_query}</div>
                            <div class="card" style="margin-top: 8px;">
                                <div class="section-title">📝 Answer</div>
                                <div class="answer-wrap">
                            """,
                            unsafe_allow_html=True
                        )
                        st.markdown(response)
                        st.markdown("</div></div>", unsafe_allow_html=True)

                        # Evidence, confidence, and source expander removed for a cleaner, chat-like response

                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")

                        # Provide fallback response
                        st.write("**🔧 Try:**")
                        st.write("- Rephrasing your question")
                        st.write("- Using simpler terms")
                        st.write("- Asking about specific sections")
            else:
                st.error("❌ PDF not processed yet. Please wait for processing to complete.")

        # Study Buddy tools
        st.markdown("---")
        st.header("🤖 AI-Powered Study Buddy")
        st.write("Explain topics in simple terms, summarize notes, or generate quizzes/flashcards on demand.")

        explain_tab, summary_tab, quiz_tab, flashcard_tab = st.tabs(
            ["🧠 Explain Simply", "📝 Summarize Notes", "❓ Generate Quiz", "🧾 Flashcards"]
        )

        with explain_tab:
            st.info("Use the main question box above to ask for explanations.")

        with summary_tab:
            if st.button("Summarize my notes", use_container_width=True, key="summary_btn"):
                if not st.session_state.processing_complete:
                    st.warning("Please wait until PDF processing finishes.")
                else:
                    with st.spinner("Generating summary..."):
                        summary = build_simple_summary(st.session_state.full_text)
                        st.markdown(summary)

        with quiz_tab:
            if st.button("Generate quiz", use_container_width=True, key="quiz_btn"):
                if not st.session_state.processing_complete:
                    st.warning("Please wait until PDF processing finishes.")
                else:
                    with st.spinner("Creating quiz questions..."):
                        pairs = extract_definition_pairs(st.session_state.full_text, max_pairs=8)
                        if len(pairs) < 3:
                            st.warning("Not enough clean definitions found. Building a comprehension quiz from key sentences instead.")
                            chunks = st.session_state.chunks or []
                            quiz = build_quiz_from_chunks(chunks, max_q=5)
                        else:
                            quiz = build_quiz_from_pairs(pairs, max_q=5)
                        st.markdown(quiz)

        with flashcard_tab:
            if st.button("Generate flashcards", use_container_width=True, key="flashcard_btn"):
                if not st.session_state.processing_complete:
                    st.warning("Please wait until PDF processing finishes.")
                else:
                    with st.spinner("Creating flashcards..."):
                        pairs = extract_definition_pairs(st.session_state.full_text, max_pairs=10)
                        if len(pairs) < 3:
                            st.warning("Not enough clean definitions found. Try using a PDF with clearer headings/definitions, or enable LLM for generation.")
                        cards = pairs[:6]
                        if not cards:
                            st.info("No flashcards available for this document.")
                        else:
                            cols = st.columns(2)
                            for i, (term, definition) in enumerate(cards, 1):
                                col = cols[(i - 1) % 2]
                                with col:
                                    st.markdown(
                                        f"""
                                        <div class="card" style="margin-bottom: 12px;">
                                            <div class="section-title">Card {i}</div>
                                            <div><strong>Front:</strong> {term}</div>
                                            <details style="margin-top: 8px;">
                                                <summary>Show answer</summary>
                                                <div style="margin-top: 6px;"><strong>Back:</strong> {definition}</div>
                                            </details>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
elif uploaded_file is None:
    st.info("👆 Please upload a PDF file to get started!")

    # Show installation status and instructions
    with st.expander("🔧 System Status & Installation Guide"):
        st.write("**📊 Vector Search Components Status:**")

        # Check FAISS
        if FAISS_AVAILABLE:
            st.success("✅ FAISS - Available (Highest performance, recommended)")
        else:
            st.error("❌ FAISS - Not installed")
            st.write("**Install FAISS for best performance:**")

            col1, col2 = st.columns(2)
            with col1:
                st.code("# For CPU (most common)\npip install faiss-cpu", language="bash")
            with col2:
                st.code("# For GPU (if you have CUDA)\npip install faiss-gpu", language="bash")

        # Check Chroma
        if CHROMA_AVAILABLE:
            st.success("✅ ChromaDB - Available (Good performance)")
        else:
            st.error("❌ ChromaDB - Not installed")
            st.code("pip install chromadb", language="bash")

        # Fallback always available
        st.info("✅ Enhanced Text Search - Always available (Basic but functional)")

        st.write("**🎯 Installation Recommendations:**")

        if not FAISS_AVAILABLE and not CHROMA_AVAILABLE:
            st.warning("**For best results, install at least one vector database:**")
            st.code(
                "# Quick install (choose one or both)\npip install faiss-cpu\n# OR\npip install chromadb\n# OR both for maximum compatibility\npip install faiss-cpu chromadb",
                language="bash")

        elif not FAISS_AVAILABLE and CHROMA_AVAILABLE:
            st.info("**ChromaDB is available. For even better performance, also install FAISS:**")
            st.code("pip install faiss-cpu", language="bash")

        elif FAISS_AVAILABLE and not CHROMA_AVAILABLE:
            st.success("**FAISS is available - you have the best performance setup!**")
            st.info("Optionally install ChromaDB as backup:")
            st.code("pip install chromadb", language="bash")

        else:
            st.success("🎉 **Perfect setup! Both FAISS and ChromaDB are available.**")

        st.write("**🔄 After Installation:**")
        st.write("1. Restart your Streamlit app")
        st.write("2. Re-upload your PDF file")
        st.write("3. Enjoy better search accuracy!")

        # Show current errors if any
        if not FAISS_AVAILABLE:
            with st.expander("🐛 FAISS Error Details"):
                error_msg = globals().get('FAISS_ERROR', 'Import failed')
                st.code(error_msg)

        if not CHROMA_AVAILABLE:
            with st.expander("🐛 ChromaDB Error Details"):
                error_msg = globals().get('CHROMA_ERROR', 'Import failed')
                st.code(error_msg)

    # Show supported PDF types
    st.markdown("""
    ### 📄 Supported PDF Types:
    - **Academic papers and research documents**
    - **Course syllabus and curriculum** 
    - **Technical manuals and guides**
    - **Reports and presentations**
    - **Books and textbooks**
    - **Scanned documents** (with readable text)
    - **Multi-page documents** of any size

    ### ✨ Features:
    - **Smart content extraction** from any PDF structure
    - **Unit/chapter-specific** question answering
    - **Topic and concept** identification
    - **Assessment and reference** information extraction
    - **Definition and procedure** explanations
    """)

# Help section
with st.expander("❓ How to Use This App"):
    st.write("""
    ### Steps:
    1. **Upload** any PDF document using the sidebar
    2. **Wait** for processing (may take a moment for large files)
    3. **Click** on suggested questions or type your own
    4. **Get** intelligent answers with source references

    ### Tips for Better Results:
    - **Use suggested questions** for best results
    - **Be specific** in your questions (e.g., "Unit 1 topics" vs "topics")
    - **Use keywords** from the document
    - **Ask follow-up questions** for clarification

    ### Common Question Types:
    - **Structural:** "What are the topics in Unit 1?"
    - **Definitions:** "What is [term]?"
    - **Lists:** "List the main topics"
    - **Procedures:** "How to [do something]?"
    - **Summary:** "What is this document about?"
    """)

# Footer
st.markdown("---")
st.markdown("📘 **PDF Chat Assistant** - Ask your documents anything!")