
# from aira.core.llm_loader import AIraModel
# from aira.chains.basic_chain import BasicChain


# def main():
#     llm = AIraModel().llm
#     chain = BasicChain(llm)

#     question = "Explain attention mechanism in simple terms."
#     response = chain.run(question)

#     print("User Question:", question)
#     print("AIra Response:", response)


# if __name__ == "__main__":
#     main()

from fastapi import FastAPI
from aira.api.chat import router as chat_router
from aira.api.health import router as health_router

app = FastAPI(
    title="AIra",
    description="Modular LLM Serving Framework",
    version="0.2.0"
)


#app.include_router(chat_router)
app.include_router(chat_router, prefix="/api/v1")
app.include_router(health_router)