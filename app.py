import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    """Clean and normalize extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Fix common OCR/extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
    text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
    text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)  # Space between letters and numbers

    # Remove extra spaces
    text = re.sub(r' +', ' ', text)

    return text.strip()


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
    """Extract definitions with better context and formatting"""
    # Try to identify what term is being asked about
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
    definition_content = []

    if term:
        # Look for sentences containing the term with definitional context
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean or len(sentence_clean) < 15:
                continue

            if term.lower() in sentence_clean.lower():
                # Check if this looks like a definition
                if any(indicator in sentence_clean.lower() for indicator in
                       [':', 'is', 'means', 'refers', 'defined as']):
                    definition_content.append(sentence_clean)
                elif len(definition_content) < 3:  # Include context
                    definition_content.append(sentence_clean)

    if definition_content:
        result = f"## Definition/Explanation:\n\n"
        result += "\n\n".join(definition_content[:6])
        return result
    else:
        # Fallback: provide general content
        meaningful_sentences = [s.strip() for s in sentences[:8] if len(s.strip()) > 20]
        result = f"## Related Information:\n\n"
        result += "\n\n".join(meaningful_sentences[:5])
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
    """Extract summary with better content selection"""
    sentences = re.split(r'[.!?]+', content)
    summary_sentences = []

    # Look for the most informative sentences
    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean or len(sentence_clean) < 20:
            continue

        # Prioritize sentences with key terms
        if any(word in sentence_clean.lower() for word in
               ['summary', 'overview', 'main', 'key', 'important', 'objective']):
            summary_sentences.insert(0, sentence_clean)  # Put at beginning
        elif len(summary_sentences) < 8:
            summary_sentences.append(sentence_clean)

    result = "## Summary/Overview:\n\n"

    if summary_sentences:
        for sentence in summary_sentences[:8]:
            result += f"• {sentence}\n\n"
    else:
        result += "No specific summary content found in the selected chunks."

    return result


def format_comprehensive_response_improved(content, query):
    """Format a comprehensive response for better presentation"""
    formatted_response = f"### Response to: {query}\n\n"
    formatted_response += content
    return formatted_response





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

    for line in lines[:50]:  # Check first 50 lines for topics
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
            topic_clean = topic.strip()
            if (len(topic_clean) > 5 and
                    len(topic_clean) < 100 and
                    topic_clean not in seen_topics and
                    len(topic_clean.split()) <= 12):
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


def generate_answer_with_hf_llm(query, chunks, model="HuggingFaceTB/SmolLM3-3B", max_tokens=300):
    """Generate an answer using Hugging Face Inference API (safe wrapper).
    `chunks` is a list of chunk-like objects with `page_content`.
    Returns the text answer or None on failure.
    """
    if not HUGGINGFACE_HUB_AVAILABLE or not InferenceClient or not HF_TOKEN:
        return None

    # Build a compact context from chunks
    try:
        context_text = "\n\n".join([c.page_content for c in chunks if hasattr(c, 'page_content')])
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nProvide a concise, factual, and well-cited answer using only the context above. If answer is not in context, say you don't know."

        client = InferenceClient(api_key=HF_TOKEN)

        # Prefer chat completion if supported
        try:
            # Some InferenceClient variants expose chat completions differently
            completion = client.chat.completions.create(model=model, messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}], max_tokens=max_tokens)
            # Extract text conservatively
            choice = completion.choices[0]
            text = getattr(choice, 'message', {}).get('content') if hasattr(choice, 'message') else (choice.get('text') if isinstance(choice, dict) else None)
            if not text:
                # Try alternate fields
                text = getattr(choice, 'text', None) or (choice.get('text') if isinstance(choice, dict) else None)
        except Exception:
            # Fallback to text generation api
            completion = client.text_generation(model=model, inputs=prompt, max_new_tokens=max_tokens)
            # Some clients return list-like structures
            if isinstance(completion, list) and completion:
                text = completion[0].get('generated_text') if isinstance(completion[0], dict) else str(completion[0])
            elif isinstance(completion, dict):
                text = completion.get('generated_text') or completion.get('text')
            else:
                text = str(completion)

        if not text:
            return None

        # Remove any internal <think>...</think> markers and trim
        clean_answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return clean_answer
    except Exception:
        return None


# Streamlit app configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📘",
    layout="wide"
)

# Main title
st.title("📘 Universal PDF Chat - Ask Your PDF Anything")

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

                # Display text extraction success
                st.success(f"✅ Successfully extracted {len(raw_text)} characters from PDF")

                # Split into chunks with optimized parameters
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,  # Slightly larger chunks for better context
                    chunk_overlap=150,  # More overlap for continuity
                    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                    length_function=len,
                    keep_separator=True
                )
                chunks = splitter.split_text(raw_text)

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
                                model_name="sentence-transformers/all-MiniLM-L6-v2"
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
                                model_name="sentence-transformers/all-MiniLM-L6-v2"
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

    # Show interface only after processing is complete
    if st.session_state.processing_complete:
        # Question interface
        st.header("💬 Ask Questions About Your PDF")

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

        # LLM options in UI: allow user to enable/disable remote LLM generation
        use_llm = st.checkbox("Use LLM-based answer generation (Hugging Face Inference)", value=False)
        llm_model = st.text_input("LLM model id (Hugging Face)", value="HuggingFaceTB/SmolLM3-3B", help="Model repo id to use on Hugging Face Inference API")
        # Debugging / tuning controls
        show_reranker_scores = st.checkbox("Show raw reranker scores (debug)", value=False)
        max_context_chars = st.number_input("Max context chars to send to LLM", min_value=500, max_value=60000, value=4000, step=100)

        # Text input for questions
        user_query = st.text_input(
            "Ask any question about your PDF:",
            value=st.session_state.current_query,
            placeholder="e.g., What are the topics in Unit 1?",
            help="Ask anything about the content of your PDF document",
            key="query_input"
        )

        # Clear the session state query after displaying it
        if st.session_state.current_query and user_query == st.session_state.current_query:
            st.session_state.current_query = ""

        # Process the query
        if user_query and user_query.strip():
            if st.session_state.vector_store or st.session_state.chunks:
                with st.spinner("🔍 Searching and analyzing..."):
                    try:
                        # Use vector search if available, otherwise use simple text search
                        if st.session_state.vector_store:
                            # Search for relevant chunks using vector similarity
                            matching_chunks = st.session_state.vector_store.similarity_search(
                                user_query,
                                k=12  # Get more chunks initially for better selection
                            )
                            # Re-rank the initial matches using embeddings + MMR for better relevance/diversity
                            try:
                                relevant_chunks = rerank_chunks_with_embeddings(matching_chunks, user_query, top_k=6)
                            except Exception:
                                # Fallback to improved heuristic scoring if reranker fails
                                relevant_chunks = find_most_relevant_chunks(matching_chunks, user_query, max_chunks=6)
                        else:
                            # Fallback to simple text search
                            relevant_chunks = simple_text_search(st.session_state.chunks, user_query, max_results=6)


                        # Generate comprehensive answer — either via local extractive logic or remote LLM
                        response = None
                        if relevant_chunks:
                            # If user opted into LLM and HF client & token are available, try LLM path first
                            if use_llm and HUGGINGFACE_HUB_AVAILABLE and HF_TOKEN:
                                try:
                                    # Optionally show reranker scores
                                    if show_reranker_scores:
                                        scored = compute_reranker_scores(relevant_chunks, user_query)
                                        with st.expander("🔎 Reranker scores"):
                                            for i, (chunk, score) in enumerate(scored[:12], 1):
                                                st.write(f"#{i} score={score:.4f} preview={chunk.page_content[:200].replace('\n',' ')}...")

                                    # Trim context to max_context_chars before sending to LLM
                                    trimmed_chunks = []
                                    total = 0
                                    for c in relevant_chunks:
                                        text = c.page_content if hasattr(c, 'page_content') else str(c)
                                        if total + len(text) > max_context_chars:
                                            # add partial if space remains
                                            remaining = max(0, max_context_chars - total)
                                            if remaining > 50:
                                                trimmed_chunks.append(type(c)(page_content=text[:remaining]) if hasattr(c, '__class__') else text[:remaining])
                                            break
                                        trimmed_chunks.append(c)
                                        total += len(text)

                                    llm_resp = generate_answer_with_hf_llm(user_query, trimmed_chunks, model=llm_model, max_tokens=300)
                                    if llm_resp:
                                        response = llm_resp
                                except Exception as e:
                                    logger.exception("LLM generation failed")
                                    response = None

                            # Fallback to extractive/rule-based answer if LLM disabled or failed
                            if not response:
                                response = generate_comprehensive_answer(user_query, relevant_chunks, st.session_state.full_text)
                        else:
                            response = "❌ No relevant content found for your query. Please try rephrasing your question or check if the content exists in the PDF."

                        # Display answer
                        st.subheader("📝 Answer:")
                        st.markdown(response)


                        # Show confidence indicator
                        if relevant_chunks:
                            # Calculate simple confidence based on number of relevant chunks found
                            confidence_score = len(relevant_chunks)

                            if confidence_score >= 4:
                                st.success("🎯 High confidence answer - Multiple relevant sources found")
                            elif confidence_score >= 2:
                                st.info("✅ Good match found - Some relevant content located")
                            else:
                                st.warning("⚠️ Limited match - Try being more specific")
                        else:
                            st.error("❌ No relevant content found")

                            # Suggest alternative queries
                            st.write("**💡 Try asking:**")
                            fallback_suggestions = generate_query_suggestions(user_query, st.session_state.full_text)
                            for suggestion in fallback_suggestions[:3]:
                                st.write(f"- {suggestion}")

                        # Show source chunks (optional)
                        if relevant_chunks:
                            with st.expander("📚 Source Content Used"):
                                for i, chunk in enumerate(relevant_chunks[:3], 1):
                                    st.write(f"**Source {i}:**")
                                    preview = chunk.page_content[:300] + "..." if len(
                                        chunk.page_content) > 300 else chunk.page_content
                                    st.write(preview)
                                    st.write("---")

                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")

                        # Provide fallback response
                        st.write("**🔧 Try:**")
                        st.write("- Rephrasing your question")
                        st.write("- Using simpler terms")
                        st.write("- Asking about specific sections")
            else:
                st.error("❌ PDF not processed yet. Please wait for processing to complete.")

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