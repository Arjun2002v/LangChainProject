"""
LangGraph Hello World (no LLM needed)

A graph has three ingredients:
  1. State  - a shared dict that flows through the graph
  2. Nodes  - plain Python functions: take the state, return updates to it
  3. Edges  - connections that decide which node runs next

Flow:  START -> greet -> shout -> END

Run:  python langgraph_basics/01_hello_world.py
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. STATE: the data shared by every node
class State(TypedDict):
    name: str
    message: str


# 2. NODES: each node receives the full state and returns only the keys it changes
def greet(state: State) -> dict:
    return {"message": f"Hello, {state['name']}!"}


def shout(state: State) -> dict:
    return {"message": state["message"].upper()}


# 3. BUILD THE GRAPH
builder = StateGraph(State)
builder.add_node("greet", greet)
builder.add_node("shout", shout)

builder.add_edge(START, "greet")   # entry point
builder.add_edge("greet", "shout")
builder.add_edge("shout", END)     # exit point

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({"name": "World", "message": ""})
    print("Final state:", result)
    print("Message:    ", result["message"])

    # stream() shows the update each node made, step by step
    print("\nStep by step:")
    for step in graph.stream({"name": "LangGraph", "message": ""}):
        print("  ", step)

    # Print the graph structure as a Mermaid diagram
    print("\nGraph diagram (Mermaid):")
    print(graph.get_graph().draw_mermaid())
