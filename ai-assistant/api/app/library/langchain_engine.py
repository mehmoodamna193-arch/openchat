"""LangChain implementation of RagEngine using ChromaDB and MMR retrieval."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .engine import RagEngine, Turn, Source, Answer, Delta, StoreInfo, RagError, IndexingError

logger = logging.getLogger(__name__)

class LangChainEngine(RagEngine):
    name = "langchain"
    label = "LangChain Enterprise RAG"
    local = False  # Uses OpenAI embeddings/LLM by default
    model = "gpt-4o"

    def __init__(self, settings):
        self.settings = settings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model=self.model, streaming=True)
        self.persist_dir = self.settings.library_data_dir / "chroma_db"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def _get_vectorstore(self, store_id: str) -> Chroma:
        # In this implementation, we use the store_id as the collection name
        return Chroma(
            collection_name=store_id,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir)
        )

    async def ensure_store(self, display_name: str) -> str:
        # For simplicity, we use the display_name as the store_id if it's valid
        # In a real system, this would be a UUID from a DB
        store_id = display_name.lower().replace(" ", "_")
        self._get_vectorstore(store_id)
        return store_id

    async def store_info(self, store_id: str) -> StoreInfo | None:
        try:
            vs = self._get_vectorstore(store_id)
            return StoreInfo(
                id=store_id,
                active=vs._collection.count(),
                embedding_model=self.embeddings.model if hasattr(self.embeddings, 'model') else "text-embedding-3-small"
            )
        except Exception:
            return None

    async def add_document(
        self,
        store_id: str,
        path: Path,
        *,
        title: str,
        mime_type: str,
        metadata: Mapping[str, Any],
    ) -> str:
        try:
            # 1. Load content
            # Simple text loading for this implementation;
            # real world would use PDFLoader/UnstructuredLoader
            content = path.read_text(encoding="utf-8", errors="ignore")

            # 2. Split content
            # Convert tokens to chars roughly (1 token ~ 4 chars)
            chunk_size = self.settings.rag_chunk_tokens * 4
            chunk_overlap = self.settings.rag_chunk_overlap * 4

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            chunks = splitter.split_text(content)
            docs = [
                Document(
                    page_content=chunk,
                    metadata={**metadata, "title": title, "doc": metadata.get("doc")}
                )
                for chunk in chunks
            ]

            # 3. Store in Chroma
            vs = self._get_vectorstore(store_id)
            ids = vs.add_documents(docs)

            # We return the path as the document id to keep it consistent with file-based tracking
            return str(path.name)
        except Exception as e:
            logger.exception(f"Failed to index {path}")
            raise IndexingError(f"Could not index {title}: {e}")

    async def delete_document(self, engine_document_id: str) -> None:
        # This requires knowing which store the doc belongs to.
        # For this MVP, we'll assume a global purge or handle it via purge()
        pass

    async def purge(self, store_id: str, doc_key: str) -> int:
        vs = self._get_vectorstore(store_id)
        # Filter by the 'doc' metadata key
        ids = vs.get(where={"doc": doc_key})["ids"]
        if ids:
            vs.delete(ids=ids)
        return len(ids) if ids else 0

    def _translate_filter(self, metadata_filter: str | None) -> dict | None:
        """
        Translates AIP-160 style strings to Chroma dict filters.
        Example: '(scope="c:1" OR scope="c:2") AND year=2026'
        -> {'$and': [{'scope': {'$in': ['c:1', 'c:2']}}, {'year': 2026}]}
        """
        if not metadata_filter:
            return None

        # This is a simplified parser for the MVP.
        # In production, we'd use a proper regex or parser.
        import re

        # Handle basic scope IN filters: (scope="v1" OR scope="v2")
        scope_match = re.search(r'\(scope="([^"]+)"\s+OR\s+scope="([^"]+)"\)', metadata_filter)
        if scope_match:
            # Extract all scope values
            scopes = re.findall(r'scope="([^"]+)"', metadata_filter)
            return {"scope": {"$in": scopes}}

        # Handle single key=value
        kv_match = re.search(r'(\w+)=("[^"]*"|\d+)', metadata_filter)
        if kv_match:
            key, val = kv_match.groups()
            val = val.strip('"')
            return {key: val}

        return None

    def stream(
        self, turns: Sequence[Turn], store_ids: Sequence[str], metadata_filter: str | None
    ) -> AsyncIterator[RagEvent]:
        # 1. Prepare Query
        user_query = turns[-1].text if turns else ""

        # 2. Retrieval (MMR)
        # We use the first store_id for this MVP
        vs = self._get_vectorstore(store_ids[0])
        chroma_filter = self._translate_filter(metadata_filter)

        # MMR retrieval
        relevant_docs = vs.max_marginal_relevance_search(
            query=user_query,
            k=self.settings.rag_top_k,
            filter=chroma_filter
        )

        sources = [
            Source(
                number=i + 1,
                title=doc.metadata.get("title", "Unknown"),
                text=doc.page_content,
                doc_key=doc.metadata.get("doc"),
            )
            for i, doc in enumerate(relevant_docs)
        ]

        context = "\n\n".join([f"[{s.number}] {s.text}" for s in sources])

        # 3. Generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided context to answer the user's question. "
                       "Always cite the source using [n] markers. If the answer isn't in the context, say you don't know.\n\n"
                       "Context:\n{context}"),
            ("user", "{query}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        full_text = ""
        # Stream the response
        for chunk in chain.stream({"context": context, "query": user_query}):
            full_text += chunk
            yield Delta(text=chunk)

        # Finally, return the Answer object
        yield Answer(
            text=full_text,
            cited_text=full_text, # In a real impl, we'd post-process citations
            sources=sources,
            model=self.model,
            grounded=len(sources) > 0
        )
