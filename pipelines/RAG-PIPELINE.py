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
from typing import List, Union, Generator, Iterator
from dotenv import load_dotenv
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core import Document, PromptTemplate, VectorStoreIndex
from llama_index.core import Document, StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from pydantic import BaseModel
import faiss
import json

from pipeline_core.arabic_processing import ArabicArticleExtractor, ChunkArticleParser, TextAnalyzer, ActorExtractor
from pipeline_utils import load_documents_with_tika, split_by_articles, fixed_width_chunking, boost_nodes_with_articles
from pipelines.pipeline_core import Chunk
from pipelines.pipeline_utils.utils import boost_nodes_with_actors, actor_matches

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.postprocessor import SentenceTransformerRerank

from pipeline_core.prompts import (
    prompt,
    ARTICLE_INSTRUCTION_TEMPLATE,
    NO_ARTICLE_INSTRUCTION,
    ACTOR_INSTRUCTION_TEMPLATE
)

from pipeline_utils.actor_utils import extract_actors_from_chunks

load_dotenv()
INDEXED_MANIFEST_PATH = os.getenv("INDEXED_MANIFEST_PATH")
PERSIST_DIR = os.getenv("PERSIST_DIR")

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
        from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
        from llama_index.core.postprocessor import SentenceTransformerRerank
        from llama_index.vector_stores.faiss import FaissVectorStore

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            request_timeout=300.0,
        )

        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-gemma",
            top_n=6,
            device="cpu",
        )

        # Load the persisted index from disk if it exists, so a server
        # restart doesn't lose everything that's already been indexed.
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            print(f"Loading persisted index from {PERSIST_DIR}")
            vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=PERSIST_DIR
            )
            self.index = load_index_from_storage(storage_context)
            print("Loaded persisted index.")
        else:
            print("No persisted index found — starting with an empty index.")
            faiss_index = faiss.IndexFlatL2(4096)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex(nodes=[], storage_context=storage_context)

    async def index_file(self):
        """
        Reads the entire ./data directory via Tika, but only chunks,
        embeds, and inserts documents whose content hash isn't already
        in the manifest. Already-indexed files are skipped after Tika
        extraction (Tika doesn't support skipping individual files).
        """
        data_dir = "./data"
        manifest = self._load_manifest()  # filename -> hash
        indexed_hashes = set(manifest.values())

        # Load existing persisted index, or build a fresh empty one
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=PERSIST_DIR
            )
            self.index = load_index_from_storage(storage_context)
        elif getattr(self, "index", None) is None:
            faiss_index = faiss.IndexFlatL2(4096)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex(nodes=[], storage_context=storage_context)

        # Pre-compute hashes for every file currently in ./data, so we know
        # which ones are new BEFORE deciding whether to bother extracting them.
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

        # One Tika pass over the whole directory, as requested
        self.documents = load_documents_with_tika(data_dir)
        print(f"Loaded docs: {len(self.documents)}")

        chunks: List[Chunk] = []

        for doc in self.documents:
            filename = doc.metadata.get("file_name")

            # Skip documents whose file isn't new — already indexed
            if filename not in new_filenames:
                continue

            split_articles = split_by_articles(doc.text)
            split_articles = [a for a in split_articles if not TextAnalyzer.is_disciplinary(a)]
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

        # One batched actor-extraction call across ALL new files' chunks
        extract_actors_from_chunks(chunks, Settings.llm)

        for chunk in chunks:
            new_document = Document(
                text=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "article_numbers": chunk.article_numbers,
                    "effective_mode": chunk.effective_mode,
                    "file_name": chunk.file_name,
                    "actors": chunk.actors,
                },
            )
            self.index.insert(new_document)

        # Mark every newly-indexed file as done
        for filename in new_filenames:
            if filename in file_hashes:
                manifest[filename] = file_hashes[filename]

        self.index.storage_context.persist(persist_dir=PERSIST_DIR)
        self._save_manifest(manifest)
        print(f"Indexed and persisted {len(new_filenames)} new file(s).")

        return report

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

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        from llama_index.core import Settings
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever

        queried_articles = ArabicArticleExtractor.extract(user_message)
        queried_actors = ActorExtractor.extract(user_message, llm=Settings.llm)

        if queried_articles:
            print("Queried articles FOUND")
            target_article = ", ".join(f"مادة {a}" for a in queried_articles)
            article_instruction = ARTICLE_INSTRUCTION_TEMPLATE.format(target_article=target_article)

            vector_retriever = self.index.as_retriever(
                similarity_top_k=20,
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="article_numbers", value=a) for a in queried_articles],
                ),
            )

            # Filter BM25 nodes to only chunks containing the queried articles
            filtered_nodes = [
                node for node in self.index.docstore.docs.values()
                if any(a in node.metadata.get("article_numbers", []) for a in queried_articles)
            ]

            if not filtered_nodes:
                print("[WARNING] No chunks matched article filter, falling back to full docstore")
                filtered_nodes = list(self.index.docstore.docs.values())

            top_k = 25

        elif queried_actors:
            print("Queried actors FOUND")
            article_instruction = ACTOR_INSTRUCTION_TEMPLATE.format(
                actors=", ".join(queried_actors)
            )

            filtered_nodes = [
                node for node in self.index.docstore.docs.values()
                if actor_matches(node.metadata.get("actors", []), queried_actors)
            ]

            if not filtered_nodes:
                print("[WARNING] No chunks matched actor filter, falling back to full docstore")
                filtered_nodes = list(self.index.docstore.docs.values())
                top_k = 25  # fallback to reasonable default since we're now searching everything
            else:
                top_k = len(filtered_nodes)

            vector_retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="actors", value=a) for a in queried_actors],
                    condition="or",
                ),
            )

        else:
            print("else statement")
            top_k = 25

            article_instruction = NO_ARTICLE_INSTRUCTION
            vector_retriever = self.index.as_retriever(similarity_top_k=top_k)

            # No filter — search everything
            filtered_nodes = list(self.index.docstore.docs.values())


        bm25_retriever = BM25Retriever.from_defaults(
            nodes=filtered_nodes,
            similarity_top_k=top_k,
        )

        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=top_k,
            num_queries=4,
            mode="reciprocal_rerank",
            use_async=False,
        )

        nodes = retriever.retrieve(user_message)
        boosted_nodes = boost_nodes_with_articles(nodes, queried_articles)
        boosted_nodes += boost_nodes_with_actors(nodes, queried_actors)

        if queried_articles:
            reranked_nodes = self.reranker.postprocess_nodes(boosted_nodes, query_str=user_message)
        else:
            reranked_nodes = boosted_nodes

        context = "\n\n".join([
            f"[المصدر: {node.metadata.get('file_name', 'غير معروف')}]\n{node.text}"
            for node in reranked_nodes
        ])

        print("\n===== RETRIEVED CHUNKS =====")
        for i, node in enumerate(reranked_nodes):
            print(f"\n--- CHUNK {i + 1} ---")
            print("METADATA:", node.metadata)
            print("TEXT:", node.text)
            print("SCORE:", node.score)
        print("===========================\n")

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