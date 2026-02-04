from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from loguru import logger
from aira.core.config import CHUNK_SIZE, CHUNK_OVERLAP

class TextChunker:
    """
    Responsible for splitting documents into smaller chunks.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        ) # splitter object created

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        """
        logger.info("Splitting documents into chunks")

        chunks = self.splitter.split_documents(documents)  #use the splitter object to split the documents

        logger.info(f"Created {len(chunks)} chunks")
        return chunks
