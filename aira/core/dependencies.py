# dependencies.py — full updated file
from typing import List
from langchain.schema import Document
from aira.core.llm_loader import AIraModel
from aira.chains.basic_chain import BasicChain
from aira.chains.rag_chain import RAGChain
from aira.rag.retriever import VectorRetriever
from aira.rag.vectorstore import FAISSVectorStore
from aira.rag.embeddings import EmbeddingModel
from aira.rag.reranker import CrossEncoderReranker
from aira.rag.bm25_retriever import BM25Retriever
from aira.rag.hybrid_retriever import HybridRetriever
from aira.core.config import FAISS_INDEX_PATH, RETRIEVER_TOP_K

_llm = None
_basic_chain = None
_rag_chain = None
_retriever = None          # raw FAISS (used internally by hybrid)
_bm25_retriever = None
_hybrid_retriever = None
_reranker = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = AIraModel().llm
    return _llm


def get_basic_chain():
    global _basic_chain
    if _basic_chain is None:
        _basic_chain = BasicChain(get_llm())
    return _basic_chain


def get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain(get_llm())
    return _rag_chain


def _get_vectorstore():
    """Loads FAISS vectorstore (shared between FAISS retriever and BM25 corpus)."""
    embedding = EmbeddingModel().get()
    vectorstore = FAISSVectorStore(embedding)
    vectorstore.load(FAISS_INDEX_PATH)
    return vectorstore


def get_faiss_retriever():
    """Raw FAISS retriever — used internally by HybridRetriever."""
    global _retriever
    if _retriever is None:
        vectorstore = _get_vectorstore()
        _retriever = VectorRetriever(vectorstore, k=RETRIEVER_TOP_K)
    return _retriever


def get_bm25_retriever():
    """BM25 retriever built from the same FAISS document corpus."""
    global _bm25_retriever
    if _bm25_retriever is None:
        vectorstore = _get_vectorstore()

        # Extract all documents stored in the FAISS index
        docstore = vectorstore.vectorstore.docstore
        all_docs = list(docstore._dict.values())

        _bm25_retriever = BM25Retriever(documents=all_docs, k=RETRIEVER_TOP_K)
    return _bm25_retriever


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def get_retriever():
    """
    Full pipeline:
    FAISS top-20 + BM25 top-20 → RRF → ~30 candidates → CrossEncoder top-5
    This is what RAG chain and agent use.
    """
    global _hybrid_retriever
    if _hybrid_retriever is None:
        faiss = get_faiss_retriever()
        bm25 = get_bm25_retriever()
        reranker = get_reranker()

        hybrid = HybridRetriever(
            faiss_retriever=faiss,
            bm25_retriever=bm25,
        )

        # Wrap hybrid with reranker
        _hybrid_retriever = _RerankedHybrid(hybrid, reranker)

    return _hybrid_retriever


class _RerankedHybrid:
    """
    Thin wrapper that chains:
    HybridRetriever → CrossEncoderReranker
    Exposes the same .retrieve(query) interface as VectorRetriever
    so all existing api/ code works unchanged.
    """

    def __init__(self, hybrid: HybridRetriever, reranker: CrossEncoderReranker):
        self.hybrid = hybrid
        self.reranker = reranker

    def retrieve(self, query: str) -> List[Document]:
        from loguru import logger
        candidates = self.hybrid.retrieve(query)
        logger.info(f"Reranking {len(candidates)} hybrid candidates...")
        final = self.reranker.rerank(query, candidates)
        logger.info(f"Final pipeline output: {len(final)} documents")
        return final