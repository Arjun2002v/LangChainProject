"""
LangGraph Step 2: conditional edges (branching)

A router function looks at the state and returns the name of the next node.

Flow:  START -> check_mood -> (happy | sad) -> END

Run:  python langgraph_basics/02_conditional_edges.py
"""

from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str
    reply: str


def check_mood(state: State) -> dict:
    # Nothing to update here; this node just exists to branch after it
    return {}


def route(state: State) -> Literal["happy", "sad"]:
    good_words = ("good", "great", "happy", "love")
    return "happy" if any(w in state["text"].lower() for w in good_words) else "sad"


def happy(state: State) -> dict:
    return {"reply": "Glad to hear it!"}


def sad(state: State) -> dict:
    return {"reply": "Sorry to hear that. Hope it gets better."}


builder = StateGraph(State)
builder.add_node("check_mood", check_mood)
builder.add_node("happy", happy)
builder.add_node("sad", sad)

builder.add_edge(START, "check_mood")
builder.add_conditional_edges("check_mood", route)  # route() picks the next node
builder.add_edge("happy", END)
builder.add_edge("sad", END)

graph = builder.compile()


if __name__ == "__main__":
    for text in ["I feel great today", "My code won't compile"]:
        result = graph.invoke({"text": text, "reply": ""})
        print(f"{text!r:28} -> {result['reply']}")
