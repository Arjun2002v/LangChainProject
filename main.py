from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain  # Note: likely a typo, should be langchain.chains

def generate_name():
    # Initialize the local Ollama LLM
    # model="llama3.2" -> uses the locally downloaded Llama 3.2 model
    # temperature=1 -> higher randomness/creativity in responses (0 = deterministic, 1 = more varied)
    llm = OllamaLLM(model="llama3.2", temperature=1)

    # Define a reusable prompt template
    # input_variables -> placeholders that get filled in dynamically when the chain runs
    # template -> the actual prompt text sent to the LLM, with {animal_type} as a placeholder
    prompt_naming_chain = PromptTemplate(
        input_variabled=['animal_type'],  # Note: likely a typo, should be input_variables
        template="Suggest a {animal_type} name"
    )

    # Directly invokes the LLM with a hardcoded prompt string
    # (this bypasses prompt_naming_chain entirely - the template above isn't actually used here)
    return llm.invoke("Suggest a product name")


if __name__ == "__main__":
    # Entry point: calls generate_name() and prints the LLM's response
    print(generate_name())