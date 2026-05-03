"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama
"""

from pathlib import Path
import os

os.environ["TIKA_SERVER_ENDPOINT"] = "http://localhost:9998"

from typing import List, Union, Generator, Iterator, Literal
from dotenv import load_dotenv
from tika import parser
from llama_index.core import Document
from pydantic import BaseModel
from llama_index.core.prompts import PromptTemplate
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
import re
from typing import List, Dict, Optional, Any
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from dataclasses import dataclass, field
import numpy as np

load_dotenv()

prompt_template = """You are an expert system analyst specializing in legal and software requirements. 
Your task is to extract functional requirements from the provided Arabic legal text chunk (e.g., "مادة 50").

Identify statements that express rules:
- Obligations (e.g., يجب, يلتزم)
- Prohibitions (e.g., لا يجوز, يُحظر)
- Permissions or rights (e.g., يجوز, يحق)

EXTRACTION RULES:
1. Translate the extracted requirement into CLEAR ARABIC.
2. Structure each requirement logically following this Arabic pattern: 
   "[الفاعل] يجب/يجوز أن [الفعل] [الشروط]" 
   (Example: "النظام يجب أن يرفض الطلب إذا كان غير مكتمل")
3. If the actor (الفاعل) is not explicitly stated, you MUST infer a reasonable system actor (e.g., "النظام" (The system), "المستخدم" (The user), "المسؤول" (The admin)).
4. Keep the output concise. Limit to the core functional requirements.
5. You MUST output ONLY valid JSON. Absolutely no conversational text, no greetings, and no explanations before or after the JSON.

{format_instructions}

Arabic Text to Analyze:
{context}
"""
#
# prompt = PromptTemplate.from_template(prompt_template)

def load_documents_with_tika(directory: str):
    documents = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            parsed = parser.from_file(file_path)
            text = parsed.get("content")

            if not text:
                continue

            # Optional: Arabic cleanup (recommended)
            text = re.sub(r"[ًٌٍَُِّْـ]", "", text).strip()

            documents.append(
                Document(
                    text=text,
                    metadata={"file_name": filename}
                )
            )

        except Exception as e:
            print(f"[TIKA ERROR] {filename}: {e}")

    return documents

class ArabicArticleExtractor:
    """prevents double-counting and handles edge cases gracefully"""

    # Arabic-Indic → ASCII digit map
    _AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    # Patterns ordered from most-specific to least-specific
    _PATTERNS: List[re.Pattern] = [
        # المواد 10 و 20 و 30  /  المواد 10، 20، 30
        re.compile(
            r"المواد\s+((?:[\d٠-٩]+\s*(?:و|،|,)\s*)+[\d٠-٩]+)",
            re.UNICODE,
        ),
        # Handles chains like:
        #   "المادة رقم 122 و المادة 12"
        #   "المادة 5 والمادة 10 والمادة 22"
        #   "المادة رقم 3 و المادة رقم 7"
        #   "المادتين 15 و 16"
        #   "مادة رقم 5"  (without ال)
        #
        # Strategy: match the FULL span (anchor + number, then any chain of
        # "و/،  [optional: المادة/مادة]  [optional: رقم]  number").
        # All digit sequences are then extracted from the whole match,
        # so every chained article number is captured regardless of whether
        # "المادة" is repeated between them.
        re.compile(
            r"(?:ال)?ماد(?:ة|تين|ات)\s+(?:رقم\s+)?[\d٠-٩]+"
            r"(?:\s*(?:و|،|,)\s*(?:(?:ال)?ماد(?:ة|تين|ات)\s+)?(?:رقم\s+)?[\d٠-٩]+)*",
            re.UNICODE,
        ),
    ]

    @classmethod
    def _normalize(cls, text: str) -> str:
        """Convert Arabic-Indic numerals to ASCII."""
        return text.translate(cls._AR_DIGITS)

    @classmethod
    def extract(cls, query: str) -> List[int]:
        """Return a sorted, deduplicated list of article numbers found in *query*."""
        query = cls._normalize(query)
        found: set[int] = set()

        for pattern in cls._PATTERNS:
            for match in pattern.finditer(query):
                # Scan the full match span for all digit sequences.
                # Pattern 0 still has a capture group for the list portion;
                # Pattern 1 is a whole-span match with no capture groups.
                # Using group(0) works correctly for both cases.
                nums = re.findall(r"\d+", match.group(0))
                found.update(int(n) for n in nums)

        return sorted(found)

# Chunk Article Parser  (run once at index-build time)
class ChunkArticleParser:
    """a focused utility to extract article numbers (المادة X) from a block of text"""

    _AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    _HEADER_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:الماد[ةة]|مادة)\s*(?:رقم\s*)?([\d٠-٩]+)",
        re.UNICODE | re.MULTILINE,
    )

    @classmethod
    def parse(cls, text: str) -> List[int]:
        text = text.translate(cls._AR_DIGITS)
        nums = [int(m.group(1)) for m in cls._HEADER_PATTERN.finditer(text)]
        return sorted(set(nums))

class QueryAnalyzer:
    """Determines whether a query is a legal article reference or a general sentence.

    The retriever operates in three modes:
        "legal"   – always treat the query as a legal article reference.
        "general" – always treat the query as a free-text / sentence query.
        "auto"    – detect automatically (default).

    Detection heuristics (fast, no model required):
        • If ArabicArticleExtractor finds at least one article number  → legal.
        • If the query contains Arabic legal keywords (مادة, المادة, قانون, …) → legal.
        • Otherwise                                                       → general.
    """

    # Keywords that strongly suggest an Arabic legal / article-reference query
    _LEGAL_KEYWORDS = re.compile(
        r"(?:الماد[ةة]|مادة|المواد|قانون|تشريع|لائحة|نظام|بند|فقرة|الفصل|الباب|القسم"
        r"|أحكام|عقوبة|جريمة|جزاء|اتفاقية|مرسوم|قرار\s+وزاري)",
        re.UNICODE,
    )

    @classmethod
    def classify(cls, query: str) -> str:
        """Return 'legal' or 'general' for the given query string."""
        if ArabicArticleExtractor.extract(query):
            return "legal"
        if cls._LEGAL_KEYWORDS.search(query):
            return "legal"
        return "general"


def extract_article_info(chunk: str):
    # matches: المادة 5, مادة (5), etc.
    match = re.search(r'(?:المادة|(?<!ال)مادة)\s*\(?(\d+)\)?', chunk)

    if match:
        article_num = match.group(1)
        return article_num

    return "Unknown"


def split_by_articles(text):
  def sanitize(text: str) -> str:
    """Remove garbage bytes and normalise whitespace."""
    # Strip null bytes
    text = text.replace("\x00", "")
    # Replace non-breaking spaces (U+00A0, U+FEFF, U+200B, etc.) with normal space
    text = re.sub(r"[\u00a0\u00ad\u200b\u200c\u200d\u200e\u200f\ufeff]", " ", text)
    # Collapse runs of spaces/tabs on the same line into a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Normalise CRLF → LF, then collapse 3+ consecutive blank lines → 2
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace from every line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()

  _ARTICLE_LINE_RE = re.compile(
    r"^[ \t]*"                          # optional leading space
    r"(?:ال)?[مM]ادة"                  # المادة / مادة
    r"(?:\s+رقم)?"                      # optional رقم
    r"\s+"
    r"(\d+)"                            # the article number  (captured)
    r"[ \t]*$",                         # nothing else on the line
    re.MULTILINE | re.UNICODE,)

  _BUND_LINE_RE = re.compile(
    r"^[ \t]*"
    r"البند"
    r"(?:\s+رقم)?"
    r"\s+"
    r"(\d+)"
    r"[ \t]*$",
    re.MULTILINE | re.UNICODE,)

  def _find_headers(text: str) -> List[dict]:
    """
    Return a sorted list of dicts: {start, end, label}
    where start/end are character offsets in *text* of the full header line
    (including its leading newline, so we can slice text neatly).
    """
    headers = []
    for pattern, prefix in ((_ARTICLE_LINE_RE, "المادة"), (_BUND_LINE_RE, "البند")):
        for m in pattern.finditer(text):
            headers.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "number": int(m.group(1)),
                    "label": f"{prefix} {m.group(1)}",
                }
            )
    # Sort by position in document
    headers.sort(key=lambda h: h["start"])
    return headers

  _SENTENCE_END_RE = re.compile(
    r"(?<=[.!?؟،؛\n])\s+",   # split after Arabic/Latin sentence-ending punctuation
    re.UNICODE,)

  def _split_sentences(block: str) -> List[str]:
    """
    Split a block of text into individual sentences, returning only
    non-empty stripped sentences.
    """
    # Also split on newline boundaries (headings, list items, etc.)
    raw = re.split(r"\n|(?<=[.!?؟،؛])\s+", block)
    out = []
    for part in raw:
        part = part.strip()
        if part:
            out.append(part)
    return out

  _NOISE_LINE_RE = re.compile(
    r"^("
    r"www\.\S+|"                             # www.something
    r"https?://\S+|"                          # URLs
    r"LAWSOCIETY\.LY.*|"                      # site watermark
    r".*lawsociety\.ly.*|"                    # site watermark variant
    r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s+[APap][Mm].*|"  # timestamps
    r"صفحة\s+\d+|"                            # page numbers (صفحة 3)
    r"[a-zA-Z0-9\-]+صفحة\s+\d+|"             # domain+صفحة
    r"REG-GPC-\S+|"                           # document reference codes
    r"المحتويات.*"                            # table of contents header
    r")$",
    re.MULTILINE | re.UNICODE,)

  def _remove_noise_lines(text: str) -> str:
    """Remove lines that are purely navigational / metadata noise."""
    cleaned = _NOISE_LINE_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned

  def parse_legal_document(raw_text: str) -> List[str]:
    """
    Accept the raw text of a legal document and return a flat list where:
      - Each article (المادة N / البند N) is a single entry containing
        the header fused with all of its body content.
      - Text before the first article (metadata, preamble, whereas clauses)
        is discarded entirely.

    Args:
        raw_text: The raw string content of the legal document.

    Returns:
        List[str]: Ordered list of article segments only.
    """
    # --- Step 1: sanitise ---
    text = sanitize(raw_text)

    # --- Step 2: remove noise ---
    text = _remove_noise_lines(text)

    # --- Step 3: locate standalone article headers ---
    headers = _find_headers(text)

    if not headers:
        # No articles at all → nothing to return
        return []

    segments: List[str] = []

    # --- Step 4: process each article (pre-article text is intentionally dropped) ---
    for i, hdr in enumerate(headers):
        # Content runs from end of this header's line to start of next header
        content_start = hdr["end"]
        content_end = headers[i + 1]["start"] if i + 1 < len(headers) else len(text)

        body = text[content_start:content_end].strip()

        # Combine header label with its body
        if body:
            entry = f"{hdr['label']} {body}"
        else:
            entry = hdr["label"]

        # Collapse excessive internal whitespace while preserving structure
        # (single newlines kept so list items / paragraphs remain readable)
        entry = re.sub(r" {2,}", " ", entry)
        entry = re.sub(r"\n{3,}", "\n\n", entry)

        segments.append(entry.strip())

    return segments
  return parse_legal_document(text)

@dataclass
class Chunk:
    chunk_id: str
    text: str
    article_numbers: List[int] = field(default_factory=list)
    effective_mode: str = None
    file_name: str = None

# 1. Fixed-Width Chunking
def fixed_width_chunking(articles, chunk_size=3, overlap=1):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(articles), step):
        chunk = articles[i:i + chunk_size]

        if chunk:  # avoid empty
            chunks.append(chunk)

    return chunks

def boost_nodes_with_articles(nodes, query):
    query_articles = ArabicArticleExtractor.extract(query)

    for node in nodes:
        node.score = node.score or 0.0

        if query_articles:
            chunk_articles = node.metadata.get("Article", [])

            if any(a in chunk_articles for a in query_articles):
                node.score += 1.0

    return nodes

class Source(BaseModel):
    document: str
    section: Optional[str]
    text_excerpt: str

class Requirement(BaseModel):
    requirement_id: str
    purpose: str
    actor: str
    action: str
    conditions: str
    priority: Literal["high", "medium", "low"] #functional req priority : high, medium, low
    # model_confidence: float #model confidence of the answer
    # source: Source #from what file it got retrieved from
    # notes: Optional[str]

class RequirementsOutput(BaseModel):
    requirements: List[Requirement]

class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
            }
        )

    # This function is called when the server is started.
    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, VectorStoreIndex

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            request_timeout=300.0,
        )

        global documents, index

        self.documents = load_documents_with_tika("./data")
        print(f"Loaded docs: {len(self.documents)}")

        # Zakaria custom chunking goes here
        #-----------------------
        for doc in self.documents:
            split_articles = split_by_articles(doc.text)
            chunks_list = fixed_width_chunking(split_articles, chunk_size=6, overlap=2)

            if not chunks_list:
                raise ValueError("chunks_list is empty — nothing to index.")

            chunks: List[Chunk] = []
            for i, article_strings in enumerate(chunks_list):
                if not isinstance(article_strings, list):
                    raise TypeError(
                        f"chunk {i} is {type(article_strings).__name__}, expected list. "
                        "Each chunk must be a list of article strings (or a list of lists of strings)."
                    )

                # Flatten one extra nesting level if articles are themselves lists
                # Supports both List[str] and List[List[str]] chunk formats
                flat: List[str] = []
                for item in article_strings:
                    if isinstance(item, list):
                        flat.extend(item)  # unwrap inner list → List[List[str]] case
                    elif isinstance(item, str):
                        flat.append(item)  # already a string → List[str] case
                    # silently skip anything else (None, empty lists, etc.)

                # Join into a single text block for embedding & article parsing
                text = "\n".join(s.strip() for s in flat if isinstance(s, str) and s.strip())
                if not text:
                    continue  # skip empty chunks entirely
                chunks.append(Chunk(chunk_id=f"chunk_{i}",
                                    text=text,
                                    article_numbers=ChunkArticleParser.parse(text),
                                    effective_mode=QueryAnalyzer.classify(text),
                                    file_name=doc.metadata.get("file_name"))
                              )

        #-----------------------

        final_chunks = []

        for chunk in chunks:
            final_chunks.append(
                Document(
                    text=chunk.text,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "article_numbers": chunk.article_numbers,
                        "effective_mode": chunk.effective_mode,
                        "file_name": chunk.file_name,
                    }
                )
            )

        print(f"Indexed {len(final_chunks)} Chunks")

        dimension = 4096

        faiss_index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        self.index = VectorStoreIndex.from_documents(
            final_chunks,
            vector_store=vector_store
        )

        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.
        from llama_index.core import Settings
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.postprocessor import SentenceTransformerRerank

        output_format = PydanticOutputParser(output_cls=RequirementsOutput)

        program = LLMTextCompletionProgram.from_defaults(
            output_parser=output_format,
            prompt=prompt,
            llm=Settings.llm,
            verbose=True,
        )

        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-gemma",
            top_n=6,
            device="cpu"
        )

        query_mode = QueryAnalyzer.classify(user_message)
        retriever = self.index.as_retriever(
            similarity_top_k=25,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="EffectiveMode", value=query_mode)
                ]
            )
        )

        bm25_retriever = BM25Retriever.from_defaults(
            nodes=list(self.index.docstore.docs.values()),
            similarity_top_k=25
        )

        retriever = QueryFusionRetriever(
            [retriever, bm25_retriever],
            similarity_top_k=3,
            num_queries=3,
            mode="reciprocal_rerank",
            use_async=False,
        )

        nodes = retriever.retrieve(user_message)
        boosted_nodes = boost_nodes_with_articles(nodes, user_message)
        reranked_nodes = reranker.postprocess_nodes(boosted_nodes, query_str=user_message)

        context = "\n\n".join([
            f"[Document: {node.metadata.get('file_name', 'unknown')}]\n{node.text}"
            for node in reranked_nodes
        ])

        print("\n===== RETRIEVED CHUNKS =====")
        for i, node in enumerate(nodes):
            print(f"\n--- CHUNK {i + 1} ---")
            print("METADATA:", node.metadata)
            print("TEXT:", node.text)
            print("SCORE:", node.score)
        print("===========================\n")

        try:
            result = program(context=context)
        except Exception as e:
            return f"Parsing error: {e}"

        return result.model_dump_json(indent=2)
