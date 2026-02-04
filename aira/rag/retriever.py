# from typing import List
# from langchain.schema import Document
# from loguru import logger

# class VectorRetriever:
#     """
#     Thin wrapper over FAISS retriever.
#     """

#     def __init__(self, vectorstore, k: int = 4):
#         self.vectorstore = vectorstore
#         self.k = k
#         self._retriever = self.vectorstore.as_retriever(
#             search_kwargs={"k": self.k}
#         )

#     def retrieve(self, query: str) -> List[Document]:
#         logger.info(f"Retrieving top-{self.k} documents for query")
#         docs = self._retriever.get_relevant_documents(query)
#         return docs
    

from typing import List
from langchain.schema import Document
from loguru import logger


class VectorRetriever:
    """
    Thin wrapper over FAISS retriever.
    """

    def __init__(self, vectorstore, k: int = 4):
        self.vectorstore = vectorstore
        self.k = k

        # IMPORTANT: call YOUR wrapper correctly
        self._retriever = self.vectorstore.as_retriever(k=self.k)

    def retrieve(self, query: str) -> List[Document]:
        logger.info(f"Retrieving top-{self.k} documents for query")
        return self._retriever.get_relevant_documents(query)
