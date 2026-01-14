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

# New V2 Retrieval Service
from rag_core.v2_hybrid_search.retrieval_service import retrieve_and_rerank

load_dotenv()

from langchain_openai import ChatOpenAI

# --- 1. Tools Init ---

@tool
def hybrid_rag_tool_v2(query: str, product_filter: str = None) -> str:
    """
    Search INTERNAL knowledge base (Star ES9 & Xiaomi SU7) using advanced Hybrid Search & DeepSeek Reranking.
    
    Args:
        query: Specific question like 'range', 'battery', 'price'.
        product_filter: Optional. 'ES9' or 'SU7'. Use this to isolate search results to a specific car model for faster and more accurate results.
    """
    print(f"DEBUG: V2 Retrieval searching for '{query}' (Filter: {product_filter})")
    try:
        # Call the new V2 service directly with filter
        return retrieve_and_rerank(query, product_filter)
    except Exception as e:
        return f"RAG V2 Error: {e}"

# --- Setup DeepSeek Model (for Think Tool / Agent Brain if needed, but here used for Agent Brain) ---
# Note: The agent main brain is Gemini-2.0-Flash below. DeepSeek is used inside retrieve_and_rerank.

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

# --- 2. Create Deep Agent ---
# Using Gemini 2.0 Flash as the main reasoning agent
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

system_prompt = """You are an advanced Meta-Cognitive Assistant for researching electric vehicles (StarEra ES9, Xiaomi SU7).
Your goal is to answer user questions comprehensively by efficiently retrieving technical details from the internal knowledge base.

**1. DATABASE & PRODUCTS**
You have access to a specialized database with detailed specs for:
- **StarEra ES9** (Metadata/Filter: "ES9") - also known as "Exeed ES9", "Star River".
- **Xiaomi SU7**  (Metadata/Filter: "SU7") - also known as "Mi EV".

**2. CRITICAL RETRIEVAL STRATEGY (V2)**
The matching performance depends heavily on the `product_filter`.
- **Single Product Query**: ALWAYS use the `product_filter` (e.g., query="battery", product_filter="ES9").
- **Comparison Query (e.g., "Compare ES9 and SU7 range")**:
  - DO NOT run a single generic search.
  - **YOU MUST CALL THE TOOL TWICE**:
    1. Call `hybrid_rag_tool_v2(query="range", product_filter="ES9")`
    2. Call `hybrid_rag_tool_v2(query="range", product_filter="SU7")`
  - This "Divide & Conquer" strategy is MUCH faster and more accurate than a single unfiltered search.

**3. META-COGNITION**
- Use the `think_tool` to plan your calls.
- Example: "User wants comparison. I will first retrieve ES9 data, then SU7 data separately."

<Task>
Your job is to use tools to gather information about the user's input topic.
You can use any of the research tools provided to you to find resources that can help answer the research question.
You can call these tools in series or in parallel, your research is conducted in a tool-calling loop.
</Task>

<Tools>
You have access to two tools:
1. `hybrid_rag_tool_v2`: For internal product data (Star ES9 & Xiaomi SU7). This tool uses Hybrid Search + DeepSeek Reranking for high precision.
2. `think_tool`: For reflection and strategic planning during research (**CRITICAL: Use think_tool after each search**)

Use them as needed to answer user questions comprehensively.

<Instructions>
Think like a human researcher with limited time. Follow these steps:
1. **Read the question carefully** - What specific information does the user need?
2. **Start with broad but specific searches** - e.g., "ES9 vs SU7 range comparison"
3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
4. **Execute narrower searches as you gather information** - Fill in the gaps
5. **Stop when you can answer confidently** - Don't keep searching for perfection
</Instructions>

<Hard Limits>
**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: Use 2-3 hybrid_rag_tool_v2 calls maximum
- **Complex queries**: Use up to 5 hybrid_rag_tool_v2 calls maximum
- **Always stop**: After 5 hybrid_rag_tool_v2 calls if you cannot find the right sources

**Stop Immediately When**:
- You can answer the user's question comprehensively
- You have 2+ relevant examples/sources for the question
- Your last 2 hybrid_rag_tool_v2 calls returned similar information
</Hard Limits>    

<Show Your Thinking>
After each hybrid_rag_tool_v2 call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I search more or provide my answer?
</Show Your Thinking>

<Final Response Format>
When providing your findings back to the orchestrator:
1. **Structure your response**: Organize findings with clear headings and detailed explanations
2. **Cite sources inline**: Use [Page X] format when referencing information from your searches
3. **Include Sources section**: End with ### Sources listing the pages used
</Final Response Format>
"""

agent = create_deep_agent(
    model=model,
    tools=[hybrid_rag_tool_v2, think_tool],
    system_prompt=system_prompt
)

# --- 3. Execute ---
if __name__ == "__main__":
    print("--- User Query: SU7和ES9的续航对比如何？ ---")
    
    # We use a stream loop to show the thought process
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "SU7和ES9的续航对比如何？"
    
    print(f"--- User Query: {query} ---")
    
    input_payload = {"messages": [{"role": "user", "content": query}]} 
    
    final_response = ""
    
    # Stream the graph execution to see step-by-step updates
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
            except Exception as e:
                print(f"Error printing update: {e}")
                print(f"Raw Update: {update}")
            print("\n\n")
