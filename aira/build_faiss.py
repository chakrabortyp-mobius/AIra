# build_faiss.py

from loguru import logger
from aira.rag.loader import PDFLoader
from aira.rag.semantic_chunker import SemanticChunker      # ← changed
from aira.rag.embeddings import EmbeddingModel
from aira.rag.vectorstore import FAISSVectorStore
from aira.core.config import FAISS_INDEX_PATH, RAG_DOC


def main():
    logger.info("Starting FAISS index build with semantic chunking")

    # Load PDFs
    loader = PDFLoader()
    documents = loader.load_directory(RAG_DOC)
    logger.info(f"Loaded {len(documents)} pages")

    # Semantic chunking — replaces fixed-size chunking
    chunker = SemanticChunker()
    chunks = chunker.split(documents)
    logger.info(f"Produced {len(chunks)} semantic chunks")

    # Embed + build FAISS
    embedding_model = EmbeddingModel().get()
    vectorstore = FAISSVectorStore(embedding_model)
    vectorstore.build(chunks)

    # Save
    vectorstore.save(FAISS_INDEX_PATH)
    logger.info(f"FAISS index saved at {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()