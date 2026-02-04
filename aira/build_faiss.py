from loguru import logger

from aira.rag.loader import PDFLoader
from aira.rag.chunker import TextChunker
from aira.rag.embeddings import EmbeddingModel
from aira.rag.vectorstore import FAISSVectorStore
from aira.core.config import FAISS_INDEX_PATH
from aira.core.config import RAG_DOC

def main():
    logger.info("Starting FAISS index build")

    #  Load PDFs
    loader = PDFLoader()
    documents = loader.load_directory(RAG_DOC)

    #  Chunk documents
    chunker = TextChunker()
    chunks = chunker.split(documents)

    #  Create embeddings
    embedding_model = EmbeddingModel().get()

    #  Build FAISS index
    vectorstore = FAISSVectorStore(embedding_model)
    vectorstore.build(chunks)

    #  Save FAISS index
    vectorstore.save(FAISS_INDEX_PATH)

    logger.info(f"FAISS index built and saved at {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
