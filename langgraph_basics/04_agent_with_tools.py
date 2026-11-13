"""
LangGraph Step 4: a basic agent (tools + loop + memory)

The LLM decides whether to call a tool. If it does, the tool runs and the
result goes back to the LLM. This repeats until the LLM answers directly.

Flow:  START -> agent -> (tools -> agent)* -> END

A checkpointer (MemorySaver) keeps the conversation per `thread_id`, so the
agent remembers earlier turns.

Needs Ollama running with:  ollama pull llama3.2
Run:  python langgraph_basics/04_agent_with_tools.py
"""

from datetime import datetime

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


# 1. TOOLS: plain functions; the docstring tells the LLM what each one does
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [add, multiply, current_time]

# 2. LLM that knows about the tools
llm = ChatOllama(model="llama3.2", temperature=0).bind_tools(tools)

SYSTEM = (
    "You are a helpful assistant. Use the tools for arithmetic and for the "
    "current time. Otherwise answer directly."
)


# 3. NODES
def agent(state: MessagesState) -> dict:
    response = llm.invoke([("system", SYSTEM)] + state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))  # runs whatever tools the LLM asked for

# 4. EDGES
builder.add_edge(START, "agent")
# tools_condition: go to "tools" if the LLM requested a tool call, else END
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")  # loop back so the LLM sees the tool result

graph = builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    config = {"configurable": {"thread_id": "demo"}}
    print("Chat with the agent (type 'quit' to exit).")
    print("Try: 'What is 12.5 times 8, plus 3?' or 'What time is it?'\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit", "q"}:
            break
        if not user:
            continue

        # stream_mode="updates" shows each node's output, so you can watch the loop
        for step in graph.stream({"messages": [("user", user)]}, config, stream_mode="updates"):
            for node, update in step.items():
                msg = update["messages"][-1]
                if node == "tools":
                    print(f"  [tool {msg.name}] -> {msg.content}")
                elif msg.tool_calls:
                    for call in msg.tool_calls:
                        print(f"  [agent calls {call['name']}({call['args']})]")
                else:
                    print(f"bot> {msg.content}\n")
