"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama
"""

import os
import sys

os.environ["TIKA_SERVER_ENDPOINT"] = "http://localhost:9998"

from typing import List, Union, Generator, Iterator
from dotenv import load_dotenv
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core import Document, PromptTemplate
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
import faiss

from pipeline_core.arabic_processing import ArabicArticleExtractor, ChunkArticleParser, QueryAnalyzer
from pipeline_core.models import Chunk, RequirementsOutput
from pipeline_utils import load_documents_with_tika, split_by_articles, fixed_width_chunking, boost_nodes_with_articles

load_dotenv()

prompt_template = """You are an expert system analyst specializing in legal and software requirements.
Your task is to extract functional requirements from the provided Arabic legal text chunk.

**IMPORTANT: The user is asking about article "{target_article}" ONLY (e.g., مادة 50).
If the text contains multiple articles, you MUST ignore all other articles and extract requirements ONLY from "{target_article}".
If "{target_article}" is not found in the text, return an empty requirements list.**

Identify statements that express rules:
- Obligations (e.g., يجب, يلتزم)
- Prohibitions (e.g., لا يجوز, يُحظر)
- Permissions or rights (e.g., يجوز, يحق)

EXTRACTION RULES:
1. Translate the extracted requirement into CLEAR ARABIC.
2. Structure each requirement logically following this Arabic pattern:
"[الفاعل] يجب/يجوز أن [الفعل] [الشروط]"
(Example: "النظام يجب أن يرفض الطلب إذا كان غير مكتمل")
3. If the actor (الفاعل) is not explicitly stated, infer a reasonable system actor.
4. Keep the output concise. Limit to the pipeline_core functional requirements.
5. You MUST output ONLY valid JSON. No conversational text before or after.

{format_instructions}

Arabic Text to Analyze:
{context}
"""

prompt = PromptTemplate(
    template=prompt_template
)


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

        self.documents = load_documents_with_tika("./data")
        print(f"Loaded docs: {len(self.documents)}")

        chunks: List[Chunk] = []

        for doc in self.documents:
            split_articles = split_by_articles(doc.text)
            chunks_list = fixed_width_chunking(split_articles, chunk_size=6, overlap=2)

            if not chunks_list:
                raise ValueError("chunks_list is empty — nothing to index.")

            for i, article_strings in enumerate(chunks_list):
                if not isinstance(article_strings, list):
                    raise TypeError(
                        f"chunk {i} is {type(article_strings).__name__}, expected list."
                    )

                flat: List[str] = []
                for item in article_strings:
                    if isinstance(item, list):
                        flat.extend(item)
                    elif isinstance(item, str):
                        flat.append(item)

                text = "\n".join(s.strip() for s in flat if isinstance(s, str) and s.strip())
                if not text:
                    continue

                chunks.append(Chunk(
                    chunk_id=f"chunk_{i}",
                    text=text,
                    article_numbers=ChunkArticleParser.parse(text),
                    effective_mode=QueryAnalyzer.classify(text),
                    file_name=doc.metadata.get("file_name"),
                ))

        final_chunks = [
            Document(
                text=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "article_numbers": chunk.article_numbers,
                    "effective_mode": chunk.effective_mode,
                    "file_name": chunk.file_name,
                }
            )
            for chunk in chunks
        ]

        print(f"Indexed {len(final_chunks)} Chunks")

        faiss_index = faiss.IndexFlatL2(4096)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        self.index = VectorStoreIndex.from_documents(
            final_chunks,
            vector_store=vector_store,
        )

    async def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
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
            device="cpu",
        )

        query_mode = QueryAnalyzer.classify(user_message)
        queried_articles = ArabicArticleExtractor.extract(user_message)
        target_article = ", ".join(f"مادة {a}" for a in queried_articles) if queried_articles else "غير محدد"

        vector_retriever = self.index.as_retriever(
            similarity_top_k=25,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="EffectiveMode", value=query_mode)]
            ),
        )

        bm25_retriever = BM25Retriever.from_defaults(
            nodes=list(self.index.docstore.docs.values()),
            similarity_top_k=25,
        )

        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
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
            result = program(context=context, target_article=target_article)
        except Exception as e:
            return f"Parsing error: {e}"

        return result.model_dump_json(indent=2)