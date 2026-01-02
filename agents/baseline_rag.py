
import os
import operator
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent
from FlagEmbedding import FlagReranker
from tavily import TavilyClient

from rag import get_vector_store, supabase

load_dotenv()

import threading

from langchain_openai import ChatOpenAI

# --- 1. Tools Init ---
vector_store = get_vector_store()
reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
reranker_lock = threading.Lock()
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def hybrid_rag_tool(query: str, product_filter: str = None) -> str:
    """
    Search INTERNAL knowledge base for product details.
    
    Args:
        query: Specific question like 'range', 'battery', 'price'
        product_filter: 'ES9' for Star ES9, 'SU7' for Xiaomi SU7. 
                       Use this to isolate search results to a specific car model.
                       HIGHLY RECOMMENDED to use this filter when you know which car you are researching.
    """
    print(f"DEBUG: Internal RAG searching for '{query}' (Filter: {product_filter})")
    query_embedding = vector_store.embeddings.embed_query(query)
    try:
        # 1. Fetch MORE candidates (50) to allow for post-filtering
        docs = supabase.rpc(
            "hybrid_search", 
            {"query_text": query, "query_embedding": query_embedding, "match_count": 50}
        ).execute().data
    except Exception as e:
        return f"RAG Error: {e}"

    if not docs: return "No internal docs found."

    # 2. Python-side Filtering (Post-Retrieval)
    if product_filter:
        filtered_docs = []
        target_keyword = ""
        if product_filter.upper() == "ES9":
            target_keyword = "ES9" 
        elif product_filter.upper() == "SU7":
            target_keyword = "SU7" 
            
        if target_keyword:
            for d in docs:
                src = d.get('metadata', {}).get('source', '')
                if target_keyword in src or target_keyword in d['content']:
                    filtered_docs.append(d)
            
            if filtered_docs:
                print(f"DEBUG: Filtered from {len(docs)} down to {len(filtered_docs)} docs for {product_filter}")
                docs = filtered_docs

    # 3. Rerank with Lock
    pairs = [[query, d['content']] for d in docs]
    
    # CRITICAL: Lock to prevent multithreading crash in FlagReranker
    with reranker_lock:
        scores = reranker.compute_score(pairs)
    
    if not isinstance(scores, list): scores = [scores]
    
    for d, s in zip(docs, scores): d['score'] = s
    top_docs = sorted(docs, key=lambda x: x['score'], reverse=True)[:5]
    
    context = "\n".join([f"- {d['content']} (Source: {d.get('metadata', {}).get('source', 'Unknown')})" for d in top_docs])
    return f"Internal Docs:\n{context}"

# --- Setup DeepSeek Model ---
# Using LangChain's ChatOpenAI adapter for DeepSeek API
model = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),
    openai_api_base='https://api.deepseek.com',
    temperature=0
)

from langchain_core.tools import tool


from langchain_openai import ChatOpenAI

# --- 2. Create Deep Agent ---
model = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),
    openai_api_base='https://api.deepseek.com',
    temperature=0
)

system_prompt = """

<Task>
Your job is to use tools to gather information about the user's input topic. 
You can call tools in series or in parallel, your research is conducted in a tool-calling loop.
</Task>         


<Tools>
You have access to:
1. `hybrid_rag_tool`: For internal product data (Star ES9 & Xiaomi SU7).
   - Parameter `product_filter`: Use "ES9" to look up Star ES9 data specifically, or "SU7" for Xiaomi SU7.
Use it as needed to answer user questions comprehensively.
</Tools>

<Instructions>
1. **Read the question carefully** - What specific information does the user need?
2. **Search for information** - Use the tool to find relevant data.
3. **Synthesize** - Combine the findings into a clear answer.
</Instructions>

## Research Limits
- Stop after 3 delegation rounds if you haven't found adequate sources
- Stop when you have sufficient information to answer comprehensively

<Final Response Format>
When providing your findings back to the orchestrator:
1. **Structure your response**: Organize findings with clear headings and detailed explanations
2. **Cite sources inline**: Use [1], [2], [3] format when referencing information from your searches
3. **Include Sources section**: End with ### Sources listing each numbered source with title and URL
</Final Response Format>
"""

agent = create_deep_agent(
    model=model,
    tools=[hybrid_rag_tool],
    system_prompt=system_prompt
)

# --- 3. Execute ---
if __name__ == "__main__":
    print("--- User Query: 星辰ES9的续航和小米相比怎么样？ ---")
    
    # We use a stream loop to show the thought process
    query = "星辰ES9的续航和小米相比怎么样？"
    
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
