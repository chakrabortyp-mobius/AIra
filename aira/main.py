from aira.core.llm_loader import AIraModel

def main():
    model = AIraModel()
    prompt = """You are a helpful assistant.
            User: Explain attention mechanism in simple terms.
            Assistant:"""
    response = model.generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
