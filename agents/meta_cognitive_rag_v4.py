import sys
import os
import operator
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent
from tavily import TavilyClient
from langchain_core.tools import tool

# New V4 Retrieval Service (BGE Only)
from rag_core.v4_bge_only.retrieval_service import RetrievalServiceV4

load_dotenv()

from langchain_openai import ChatOpenAI

# Initialize service globally to reuse connections and model loading
try:
    v4_service = RetrievalServiceV4()
except Exception as e:
    print(f"Failed to init V4 Service: {e}")
    v4_service = None

# --- 1. Tools Init ---

@tool
def hybrid_rag_tool_v4(query: str, product_filter: str = None) -> str:
    """
    Search INTERNAL knowledge base (Star ES9 & Xiaomi SU7) using Hybrid Search & BGE-M3 Reranking.
    
    Args:
        query: Specific question like 'range', 'battery', 'price'.
        product_filter: Optional. 'ES9' or 'SU7'. Use this to isolate search results.
    """
    print(f"DEBUG: V4 BGE Retrieval searching for '{query}' (Filter: {product_filter})")
    try:
        if v4_service:
            return v4_service.search(query, product_filter)
        else:
            return "Error: Retrieval Service V4 not initialized."
    except Exception as e:
        return f"RAG V4 Error: {e}"

# --- Setup for Agent ---

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.

    Args:
        reflection: Analysis of search results and plan for next steps.
    """
    return f"Reflection recorded: {reflection}"

# --- 2. Create Deep Agent ---
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

system_prompt = """You are an advanced Meta-Cognitive Assistant (V4 Experimental) for researching electric vehicles.
Your goal is to answer user questions comprehensively by efficiently retrieving technical details from the internal knowledge base.

**1. DATABASE & PRODUCTS**
- **StarEra ES9** (Metadata/Filter: "ES9")
- **Xiaomi SU7**  (Metadata/Filter: "SU7")

**2. CRITICAL RETRIEVAL STRATEGY**
- **Single Product Query**: ALWAYS use the `product_filter`.
- **Comparison Query**: "Divide & Conquer".
    1. Call `hybrid_rag_tool_v4(query="...", product_filter="ES9")`
    2. Call `hybrid_rag_tool_v4(query="...", product_filter="SU7")`

**3. META-COGNITION**
- Use the `think_tool` to plan your calls.
- After each search, assess if you have enough information.

<Task>
Your job is to use tools to gather information about the user's input topic.
</Task>

<Tools>
1. `hybrid_rag_tool_v4`: Internal product data. Uses Hybrid Search + BGE-M3 Reranking (Local Model).
2. `think_tool`: For reflection and planning.
</Tools>

<Instructions>
1. **Read the question carefully**.
2. **Start with broad but specific searches**.
3. **Execute narrower searches** to fill gaps.
4. **Stop when you can answer confidently**.
</Instructions>

<Hard Limits>
- **Simple queries**: 2-3 calls max.
- **Complex queries**: 5 calls max.
</Hard Limits>    

<Final Response Format>
1. **Structure your response**: Clear headings.
2. **Cite sources inline**: [Page X].
3. **Include Sources section**.
</Final Response Format>
"""

agent = create_deep_agent(
    model=model,
    tools=[hybrid_rag_tool_v4, think_tool],
    system_prompt=system_prompt
)

# --- 3. Execute ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "SU7和ES9的续航对比如何？"
    
    print(f"--- User Query: {query} ---")
    
    input_payload = {"messages": [{"role": "user", "content": query}]} 
    
    for chunk in agent.stream(input_payload):
        for node, update in chunk.items():
            print(f"--- Update from node: {node} ---")
            try:
                if isinstance(update, dict) and "messages" in update and update["messages"]:
                    last_msg = update["messages"][-1]
                    try:
                        last_msg.pretty_print()
                    except:
                        print(last_msg)
                else:
                    print(f"Update content (Raw): {update}")
            except:
                pass
