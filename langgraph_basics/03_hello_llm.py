"""
LangGraph Step 3: a node that calls an LLM (local Ollama)

Uses MessagesState, a built-in state with a `messages` list.
New messages returned by a node are APPENDED, not overwritten.

Needs Ollama running with:  ollama pull llama3.2
Run:  python langgraph_basics/03_hello_llm.py
"""

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END

llm = ChatOllama(model="llama3.2", temperature=0.7)


def chatbot(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: allow emoji in LLM replies

    result = graph.invoke({"messages": [("user", "Say hello world in a fun way.")]})
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}")
