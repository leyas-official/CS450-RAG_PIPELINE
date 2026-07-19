"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 2.0
license: MIT
description: A NotebookLM-style pipeline for querying Arabic legal documents
             with a focus on functional requirements extraction.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama
"""

import os
import hashlib
from typing import List, Union, Generator, Iterator, Tuple
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
import chromadb
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel
import json

from pipeline_core.arabic_processing import ArabicArticleExtractor, ChunkArticleParser, TextAnalyzer, ActorExtractor
from pipeline_utils import load_documents_with_tika, split_by_articles, fixed_width_chunking
from pipelines.pipeline_core import Chunk
# from pipelines.pipeline_utils.utils import boost_nodes_with_actors, boost_nodes_with_articles
from pipelines.pipeline_utils.actor_utils import actor_matches

from llama_index.core import Settings
from llama_index.core.postprocessor import SentenceTransformerRerank

from pipeline_core.prompts import (
    prompt,
    ARTICLE_INSTRUCTION_TEMPLATE,
    NO_ARTICLE_INSTRUCTION,
    ACTOR_INSTRUCTION_TEMPLATE
)

from dotenv import find_dotenv

# find_dotenv() (usecwd=False, the default) walks UP from this file's own
# directory looking for a .env — this is cwd-independent. It's needed here
# because .env lives at the project root (one level above this file's own
# folder), not next to RAG_PIPELINE.py itself.
_ENV_PATH = find_dotenv(usecwd=False)
load_dotenv(_ENV_PATH)

# Resolve relative PERSIST_DIR / INDEXED_MANIFEST_PATH against the .env's
# own directory (NOT this file's directory, and NOT the process's cwd).
# Otherwise, a relative path in .env silently points at a different (often
# empty) location depending on where the importing script is run from.
_BASE_DIR = os.path.dirname(_ENV_PATH) if _ENV_PATH else os.path.dirname(os.path.abspath(__file__))


def _resolve_path(value: str | None) -> str | None:
    if not value:
        return value
    return value if os.path.isabs(value) else os.path.join(_BASE_DIR, value)


INDEXED_MANIFEST_PATH = _resolve_path(os.getenv("INDEXED_MANIFEST_PATH"))
PERSIST_DIR = _resolve_path(os.getenv("PERSIST_DIR"))

print(f"[RAG_PIPELINE] .env found at: {_ENV_PATH}")
print(f"[RAG_PIPELINE] PERSIST_DIR resolved to: {PERSIST_DIR}")
print(f"[RAG_PIPELINE] INDEXED_MANIFEST_PATH resolved to: {INDEXED_MANIFEST_PATH}")


class NoIndexedFilesError(ValueError):
    """Raised when the vector store has no indexed nodes at all."""
    pass


class NoMatchingFilesError(ValueError):
    """Raised when `selected_files` doesn't match any indexed node."""
    pass


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None
        self.reranker = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
            }
        )

    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, VectorStoreIndex, StorageContext

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            request_timeout=300.0,
        )

        Settings.extraction_llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            request_timeout=300.0,
            temperature=0.0,
        )

        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-gemma",
            top_n=6,
            device="cpu",
        )

        Settings.chunk_size = 4096

        chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_collection = chroma_client.get_or_create_collection("RAG_PIPELINE")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    async def index_file(self):
        data_dir = "./data"
        manifest = self._load_manifest()
        indexed_hashes = set(manifest.values())

        file_hashes = {}
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if not os.path.isfile(file_path):
                continue
            with open(file_path, "rb") as f:
                file_hashes[filename] = hashlib.sha256(f.read()).hexdigest()

        new_filenames = {
            filename for filename, h in file_hashes.items()
            if h not in indexed_hashes
        }

        report = []

        if not new_filenames:
            print("No new files found in ./data.")
            return report

        self.documents = load_documents_with_tika(data_dir)
        print(f"Loaded docs: {len(self.documents)}")

        chunks: List[Chunk] = []

        for doc in self.documents:
            filename = doc.metadata.get("file_name")
            if filename not in new_filenames:
                continue

            split_articles = split_by_articles(doc.text)
            # split_articles = [a for a in split_articles if not TextAnalyzer.is_disciplinary(a)]
            chunks_list = fixed_width_chunking(split_articles, chunk_size=1, overlap=0)

            if not chunks_list:
                report.append({"filename": filename, "status": False, "detail": "No usable chunks extracted."})
                continue

            file_chunks = []
            for i, article_strings in enumerate(chunks_list):
                if not isinstance(article_strings, list):
                    raise TypeError(f"chunk {i} is {type(article_strings).__name__}, expected list.")

                flat: List[str] = []
                for item in article_strings:
                    if isinstance(item, list):
                        flat.extend(item)
                    elif isinstance(item, str):
                        flat.append(item)

                text = "\n".join(s.strip() for s in flat if isinstance(s, str) and s.strip())
                if not text:
                    continue

                file_chunks.append(Chunk(
                    chunk_id=f"{filename}_chunk_{i}",
                    text=text,
                    article_numbers=ChunkArticleParser.parse(text),
                    effective_mode=TextAnalyzer.classify(text),
                    file_name=filename,
                ))

            if not file_chunks:
                report.append({"filename": filename, "status": False, "detail": "No usable chunks extracted."})
                continue

            chunks.extend(file_chunks)
            report.append({"filename": filename, "status": True, "detail": f"Indexed {len(file_chunks)} chunk(s)."})

        if not chunks:
            print("No usable chunks among new files.")
            return report

        # Extract actors per chunk using ActorExtractor — the SAME extractor
        # (same prompt, same LLM call shape) used for queries in retrieve().
        # Previously this used a separate function, extract_actors_from_chunks,
        # with its own independent prompt. Having chunk-side and query-side
        # extraction go through two different prompts meant they could easily
        # drift into different phrasing for the same real-world actor (e.g.
        # one saying "اللجنة الشعبية العامة", the other "أمين اللجنة الشعبية
        # العامة للضمان الاجتماعي" for what's conceptually the same entity),
        # which is part of why actor matching was unreliable. Using one
        # extractor for both sides doesn't fully solve that on its own (see
        # actor_matches' embedding-based rewrite for the matching side), but
        # it at least removes one avoidable source of inconsistency.
        print(f"Extracting actors from {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            chunk.actors = ActorExtractor.extract(chunk.text, llm=Settings.extraction_llm)
            print(f"  [{i + 1}/{len(chunks)}] {chunk.chunk_id} → actors: {chunk.actors}")
        print("Actor extraction complete.")

        for chunk in chunks:
            new_document = Document(
                text=chunk.text,
                doc_id=chunk.chunk_id,  # needed for deletion later
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "article_numbers": json.dumps(chunk.article_numbers),
                    "effective_mode": chunk.effective_mode,
                    "file_name": chunk.file_name,
                    "actors": json.dumps(chunk.actors),
                },
            )
            self.index.insert(new_document)

        for filename in new_filenames:
            if filename in file_hashes:
                manifest[filename] = file_hashes[filename]

        self._save_manifest(manifest)
        print(f"Indexed {len(new_filenames)} new file(s).")

        return report

    def remove_file(self, filename: str) -> int:
        if self.index is None:
            return 0

        vector_store = self.index.vector_store

        nodes = vector_store.get_nodes(
            node_ids=None,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="file_name", value=filename)]
            ),
        )

        node_ids = [node.node_id for node in nodes]
        removed_count = len(node_ids)

        print(f"DEBUG: found {removed_count} chunks for {filename}")

        if removed_count > 0:
            vector_store.delete_nodes(node_ids=node_ids)

            manifest = self._load_manifest()
            if filename in manifest:
                del manifest[filename]
                self._save_manifest(manifest)

        return removed_count

    @staticmethod
    def _load_manifest() -> dict:
        if os.path.exists(INDEXED_MANIFEST_PATH):
            with open(INDEXED_MANIFEST_PATH, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save_manifest(manifest: dict):
        os.makedirs(PERSIST_DIR, exist_ok=True)
        with open(INDEXED_MANIFEST_PATH, "w") as f:
            json.dump(manifest, f)

    def is_already_indexed(self, file_hash: str) -> bool:
        manifest = self._load_manifest()
        return file_hash in manifest.values()

    async def on_shutdown(self):
        pass

    def retrieve(self, user_message: str, selected_files: List[str] = None) -> Tuple[List, str]:
        """
        Isolated retrieval logic (previously inlined in `pipe()`).

        Loads all indexed nodes into the docstore, applies file/article/actor
        scoping, runs hybrid vector+BM25 fusion retrieval, boosts, and
        reranks. Returns the ranked nodes plus the instruction string to use
        in the final prompt.

        This is split out from `pipe()` so it can be:
          - reused by pipe() for the live app
          - called directly by offline evaluation harnesses (e.g. RAGAS)
            to test retrieval quality in isolation, without needing an LLM
            call at the end.

        Args:
            user_message: the user's query.
            selected_files: optional list of file names to scope retrieval to.

        Returns:
            (reranked_nodes, article_instruction)

        Raises:
            NoIndexedFilesError: if the vector store has no indexed nodes.
            NoMatchingFilesError: if selected_files doesn't match any node.
        """
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever

        all_nodes = self.index.vector_store.get_nodes(node_ids=None, filters=None)
        if not all_nodes:
            raise NoIndexedFilesError("No files are currently indexed.")

        self.index.docstore.add_documents(all_nodes)

        selected_files = selected_files or []
        if selected_files:
            scoped_nodes = [
                node for node in self.index.docstore.docs.values()
                if node.metadata.get("file_name") in selected_files
            ]
            if not scoped_nodes:
                raise NoMatchingFilesError("Selected files are not indexed or contain no searchable data.")
        else:
            scoped_nodes = list(self.index.docstore.docs.values())

        queried_articles = ArabicArticleExtractor.extract(user_message)
        queried_actors = ActorExtractor.extract(user_message, llm=Settings.extraction_llm)

        if queried_articles:
            print("Queried articles route.")
            target_article = ", ".join(f"مادة {a}" for a in queried_articles)
            article_instruction = ARTICLE_INSTRUCTION_TEMPLATE.format(target_article=target_article)

            filtered_nodes = [
                node for node in scoped_nodes
                if any(a in json.loads(node.metadata.get("article_numbers", "[]")) for a in queried_articles)
            ]

            if not filtered_nodes:
                # No chunk's parsed article_numbers matched the queried article(s)
                # (could be a format mismatch, a typo, or an article not present
                # in this document set). Fall back to the full scoped set rather
                # than handing BM25/vector retrievers an empty node list.
                print(f"[WARNING] No chunks matched article filter {queried_articles}, "
                      f"falling back to full scoped set ({len(scoped_nodes)} nodes)")
                filtered_nodes = scoped_nodes

        elif queried_actors:
            print("Queried actors route.")
            article_instruction = ACTOR_INSTRUCTION_TEMPLATE.format(
                actors=", ".join(queried_actors)
            )

            filtered_nodes = [
                node for node in scoped_nodes
                if actor_matches(json.loads(node.metadata.get("actors", [])), queried_actors)
            ]

            if not filtered_nodes:
                print(f"[WARNING] No chunks matched actor filter {queried_actors}, "
                      f"falling back to full scoped set ({len(scoped_nodes)} nodes)")
                filtered_nodes = scoped_nodes

        else:
            print("else statement route.")
            article_instruction = NO_ARTICLE_INSTRUCTION
            filtered_nodes = scoped_nodes

        if not filtered_nodes:
            # Only reachable if scoped_nodes itself was empty going in.
            raise NoMatchingFilesError("No indexed chunks available for this query scope.")

        top_k = min(25, len(filtered_nodes))
        filtered_index = VectorStoreIndex(filtered_nodes)
        vector_retriever = filtered_index.as_retriever(
            similarity_top_k=top_k
        )

        bm25_retriever = BM25Retriever.from_defaults(
            nodes=filtered_nodes,
            similarity_top_k=top_k,
        )

        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=top_k,
            num_queries=4,
            mode="reciprocal_rerank",
            use_async=False,
        )

        nodes = fusion_retriever.retrieve(user_message)

        # nodes = boost_nodes_with_articles(nodes, queried_articles)
        # nodes = boost_nodes_with_actors(nodes, queried_actors)
        # boosted_nodes = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
        #
        # if queried_articles:
        #     reranked_nodes = self.reranker.postprocess_nodes(boosted_nodes, query_str=user_message)
        # else:
        #     reranked_nodes = boosted_nodes

        reranked_nodes = self.reranker.postprocess_nodes(nodes, query_str=user_message)

        return reranked_nodes, article_instruction

    # model_id: str, messages: List[dict],
    # just in case if I need these parameters

    def pipe(
        self, user_message: str, body: dict, model_id: str, messages: List[dict],
    ) -> Union[str, Generator, Iterator]:
        selected_files = body.get("selected_files") or []

        try:
            reranked_nodes, article_instruction = self.retrieve(user_message, selected_files=selected_files)
        except NoIndexedFilesError:
            return "لا توجد ملفات مفهرسة حاليًا. يرجى رفع ملف واحد على الأقل قبل البدء بطرح الأسئلة."
        except NoMatchingFilesError:
            return "الملفات المحددة غير مفهرسة أو لا تحتوي على بيانات قابلة للبحث."

        context_parts = []

        for i, node in enumerate(reranked_nodes, start=1):
            file_name = node.metadata.get('file_name', 'غير معروف')
            article_numbers = json.loads(node.metadata.get('article_numbers', '[]'))
            actors = json.loads(node.metadata.get('actors', '[]'))
            effective_mode = node.metadata.get('effective_mode', '')

            header = f"[المصدر: {file_name}]"
            if article_numbers:
                header += f" [مادة: {', '.join(str(a) for a in article_numbers)}]"
            if actors:
                header += f" [الفاعل المحتمل: {', '.join(actors)}]"
            if effective_mode:
                header += f" [الوضع: {effective_mode}]"

            chunk = f"{header}\n{node.text}"

            print(f"\n{'=' * 80}")
            print(f"CHUNK {i}")
            print(f"{'=' * 80}")
            print(chunk)

            context_parts.append(chunk)

        context = "\n\n".join(context_parts)

        formatted_prompt = prompt.format(
            article_instruction=article_instruction,
            user_message=user_message,
            context=context,
        )

        try:
            response = Settings.llm.complete(formatted_prompt)
            return response.text
        except Exception as e:
            return f"خطأ أثناء المعالجة: {e}"