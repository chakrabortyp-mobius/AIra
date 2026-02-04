from aira.core.llm_loader import AIraModel
from aira.chains.basic_chain import BasicChain
from aira.chains.rag_chain import RAGChain
from aira.rag.retriever import VectorRetriever
from aira.rag.vectorstore import FAISSVectorStore
from aira.rag.embeddings import EmbeddingModel

from aira.core.config import FAISS_INDEX_PATH 

_llm = None
_basic_chain = None
_rag_chain = None
_retriever = None


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


def get_retriever():
    global _retriever
    if _retriever is None:
        embedding = EmbeddingModel().get()
        vectorstore = FAISSVectorStore(embedding)
        vectorstore.load(FAISS_INDEX_PATH )
        _retriever = VectorRetriever(vectorstore)
    return _retriever
