
import sys
import os
from langgraph.checkpoint.memory import MemorySaver

# Add project root to sys.path
sys.path.append(os.getcwd())

# Re-import everything needed to recreate the agent with memory
from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent
from agents.meta_cognitive_rag_v2 import hybrid_rag_tool_v2, think_tool, system_prompt

def test_memory():
    # 1. Setup Model & Agent
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    # Enable Memory
    memory = MemorySaver()
    
    # Re-create agent properly passing the checkpointer
    agent_graph = create_deep_agent(
        model=model,
        tools=[hybrid_rag_tool_v2, think_tool],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    # 2. Define Thread Config
    config = {"configurable": {"thread_id": "test_user_1"}}

    # 3. Turn 1: "I am Tom"
    print("\n--- TURN 1: 'My name is Tom' ---")
    input_1 = {"messages": [{"role": "user", "content": "Hello, my name is Tom."}]}
    
    # Run to completion
    for text in agent_graph.stream(input_1, config=config):
        pass # Just consume stream to update state
        
    print("(Turn 1 finished)")

    # 4. Turn 2: "Who am I?"
    print("\n--- TURN 2: 'Who am I?' ---")
    input_2 = {"messages": [{"role": "user", "content": "Who am I?"}]}
    
    final_answer = ""
    for chunk in agent_graph.stream(input_2, config=config):
        for node, update in chunk.items():
            try:
                # deepagents might return a custom object or dict. 
                # If it's a dict:
                if isinstance(update, dict) and "messages" in update:
                    msg = update["messages"][-1]
                    if hasattr(msg, 'content'):
                        final_answer = msg.content
                # If it's behavior from langgraph prebuilt (sometimes returns list or other structures)
                # Let's just print to safely see whatever we got
                # print(f"DEBUG node={node} type={type(update)}")
            except Exception as e:
                print(f"Skipping chunk due to error: {e}")

    print(f"\n🤖 V2 Answer: {final_answer}")

if __name__ == "__main__":
    test_memory()
