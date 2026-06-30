from langchain_ollama import OllamaLLM

def generate_name():
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    return llm.invoke("Suggest a product name")

if __name__ == "__main__":
    print(generate_name())