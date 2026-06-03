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
from typing import List, Union, Generator, Iterator
from dotenv import load_dotenv
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core import Document, PromptTemplate
from pydantic import BaseModel
import faiss

from pipeline_core.arabic_processing import ArabicArticleExtractor, ChunkArticleParser, TextAnalyzer, ActorExtractor
from pipeline_utils import load_documents_with_tika, split_by_articles, fixed_width_chunking, boost_nodes_with_articles
from pipelines.pipeline_core import Chunk
from pipelines.pipeline_utils.utils import boost_nodes_with_actors, actor_matches

load_dotenv()

prompt_template = """أنت محلل متطلبات برمجية متخصص في استخلاص المتطلبات الوظيفية من الوثائق القانونية والتشريعية العربية،
بهدف تحويلها إلى متطلبات نظام قابلة للتنفيذ من قِبل فريق هندسة البرمجيات.

{article_instruction}

**تعريف المتطلب الوظيفي في سياق هندسة البرمجيات:**
المتطلب الوظيفي هو وصف لما يجب أن يفعله النظام البرمجي بشكل مباشر وقابل للتنفيذ.
كل متطلب يجب أن يكون قابلاً للترجمة إلى كود، endpoint، قاعدة عمل، أو تدفق بيانات.

- **الالتزامات (Obligations):** ما يجب أن ينفذه النظام تلقائياً أو عند حدث معين
  مستخلص من: يجب، يلتزم، يتعين
  أمثلة برمجية:
  - النظام يجب أن يرسل إشعاراً تلقائياً عند انتهاء صلاحية الترخيص
  - النظام يجب أن يحفظ سجل بكل عملية تقديم مع timestamp
  - النظام يجب أن يتحقق من اكتمال الحقول الإلزامية قبل قبول الطلب

- **القيود (Constraints):** ما يجب أن يرفضه أو يمنعه النظام على مستوى validation أو access control
  مستخلص من: لا يجوز، يُحظر، يُمنع
  أمثلة برمجية:
  - النظام يجب أن يمنع تقديم طلب جديد إذا كان هناك طلب معلق لنفس المستخدم
  - النظام يجب أن يرفض رفع ملف يتجاوز الحجم المحدد أو بصيغة غير مدعومة
  - النظام يجب أن يحجب الوصول إذا لم تكتمل شروط التسجيل

- **الصلاحيات (Permissions/Rights):** ما يتيحه النظام لأدوار معينة على مستوى role-based access
  مستخلص من: يجوز، يحق، يُتيح
  أمثلة برمجية:
  - المدير فقط يملك صلاحية الاطلاع على تقارير جميع المستخدمين
  - يُتاح للمتقدم تعديل بياناته فقط قبل إغلاق باب التقديم
  - يحق للجهة المختصة إلغاء الطلب وإعادته للمراجعة في أي مرحلة

- **الشروط (Conditions):** منطق if/then قابل للتنفيذ كـ business rules أو workflow transitions
  مستخلص من: إذا، في حال، متى، بشرط
  أمثلة برمجية:
  - إذا تجاوزت مدة المراجعة 30 يوماً → يُصدر النظام تنبيهاً تلقائياً للمسؤول
  - في حال رفض الطلب → يجب أن يُسجّل النظام سبب الرفض ويُرسله للمتقدم
  - إذا لم يستوفِ المتقدم الشروط → يُوقف النظام تقدمه في الخطوة التالية

**ما لا يُعدّ متطلباً وظيفياً قابلاً للبرمجة — تجاهله تماماً:**
هذه المتطلبات تخص السلوك الإنساني والانضباط المؤسسي ولا يمكن ترجمتها إلى كود برمجي:
- الالتزامات الأخلاقية والسلوكية (مثل: يجب أن يتحلى الموظف بالأمانة)
- قواعد الانضباط الوظيفي (مثل: يُحظر على الموظف الإخلال بالنظام)
- العقوبات والجزاءات التأديبية (مثل: يُعاقب على المخالفة بالفصل أو الغرامة)
- الالتزامات الأخلاقية تجاه الزملاء أو المجتمع
- أي متطلب لا يمكن ترجمته إلى: واجهة مستخدم، قاعدة بيانات، API، قاعدة عمل برمجية، أو تدفق نظام

**اسأل نفسك قبل إدراج أي متطلب: هل يمكن برمجة هذا في نظام معلومات؟**
- إذا كانت الإجابة نعم → أدرجه
- إذا كانت الإجابة لا → تجاهله تماماً ولا تذكره

**تعليمات الإجابة:**
- أجب على سؤال المستخدم بالعربية الفصحى من منظور هندسة المتطلبات البرمجية.
- لكل متطلب وظيفي تستخلصه، اذكر:
  - **المصدر (Source):** رقم المادة التي استُخلص منها المتطلب (مثل: المادة 5، المادة 12)
  - **الفاعل (Actor):** من يقوم بالفعل أو من يتأثر به (مستخدم، نظام، جهة خارجية)
  - **الفعل (Action):** ما الذي يجب أن يحدث
  - **الشرط (Condition):** متى أو في أي حالة يُطبَّق هذا المتطلب (إن وُجد)
- إذا كان المتطلب مستخلصاً من أكثر من مادة، اذكر جميع المواد المعنية.
- استند فقط إلى المقاطع المسترجعة أدناه، ولا تستنتج متطلبات غير موجودة في النص.
- إذا لم تجد إجابة كافية في النصوص المقدمة، صرّح بذلك بوضوح.

**سؤال المستخدم:**
{user_message}

**المقاطع المسترجعة من الوثائق:**
{context}
"""

prompt = PromptTemplate(template=prompt_template)

ARTICLE_INSTRUCTION_TEMPLATE = """**تنبيه:** المستخدم يسأل عن {target_article} تحديداً.
ركّز إجابتك على ما يخص هذه المادة/المواد فقط، وتجاهل المحتوى غير المرتبط بها."""

NO_ARTICLE_INSTRUCTION = """**نطاق البحث:** السؤال عام ويشمل جميع الوثائق المتاحة.
أجب بناءً على أكثر المقاطع صلةً بالسؤال."""

ACTOR_INSTRUCTION_TEMPLATE = """**تنبيه:** المستخدم يسأل عن الفاعل/الطرف: {actors}.
ركّز إجابتك على المواد التي يظهر فيها هذا الطرف، واستخرج المتطلبات الوظيفية المرتبطة به تحديداً."""


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
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.core.postprocessor import SentenceTransformerRerank

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

        # Documents are already loaded via Tika from an external source
        self.documents = load_documents_with_tika("./data")
        print(f"Loaded docs: {len(self.documents)}")

        chunks: List[Chunk] = []

        for doc in self.documents:
            split_articles = split_by_articles(doc.text)
            split_articles = [a for a in split_articles if not TextAnalyzer.is_disciplinary(a)]
            chunks_list = fixed_width_chunking(split_articles, chunk_size=1, overlap=0)

            if not chunks_list:
                raise ValueError("chunks_list is empty — nothing to index.")

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

                chunks.append(Chunk(
                    chunk_id=f"chunk_{i}",
                    text=text,
                    article_numbers=ChunkArticleParser.parse(text),
                    effective_mode=TextAnalyzer.classify(text),
                    file_name=doc.metadata.get("file_name"),
                ))

        # --- Actor extraction pass ---
        actor_extraction_prompt = """أنت محلل نصوص قانونية. مهمتك استخراج الفاعلين (الأطراف) من النص القانوني التالي.
        الفاعل هو أي طرف يُذكر كمنفذ لفعل أو مُلزَم بالتزام أو صاحب حق.
        أمثلة: الطالب، المرخص له، الجهة المختصة، المؤسسة، المدير، الموظف.

        أعد قائمة الفاعلين فقط، كل فاعل في سطر مستقل، بدون ترقيم أو شرح.
        إذا لم يوجد فاعل واضح، أعد كلمة: لا يوجد

        النص:
        {text}"""

        print(f"Extracting actors from {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            try:
                response = Settings.llm.complete(
                    actor_extraction_prompt.format(text=chunk.text)
                )
                raw = response.text.strip()

                if "لا يوجد" in raw:
                    chunk.actors = []
                else:
                    chunk.actors = [
                        line.strip()
                        for line in raw.splitlines()
                        if line.strip()
                    ]

                print(f"  [{i + 1}/{len(chunks)}] {chunk.chunk_id} → actors: {chunk.actors}")

            except Exception as e:
                print(f"  [ERROR] {chunk.chunk_id}: {e}")
                chunk.actors = []

        print("Actor extraction complete.")
        # ----------------------------

        final_chunks = [
            Document(
                text=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "article_numbers": chunk.article_numbers,
                    "effective_mode": chunk.effective_mode,
                    "file_name": chunk.file_name,
                    "actors": chunk.actors,
                }
            )
            for chunk in chunks
        ]

        faiss_index = faiss.IndexFlatL2(4096)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        self.index = VectorStoreIndex.from_documents(
            documents=final_chunks,
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

        queried_articles = ArabicArticleExtractor.extract(user_message)
        queried_actors = ActorExtractor.extract(user_message, llm=Settings.llm)

        tok_k = 0
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