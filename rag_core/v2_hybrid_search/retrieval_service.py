import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from supabase import create_client, Client

load_dotenv()

# --- Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") # Use Service Key for RLS bypass if needed, or Anon
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Initialize DeepSeek LLM for Reranking
# Assuming DeepSeek API is OpenAI compatible
llm_reranker = ChatOpenAI(
    model="deepseek-chat", # or deepseek-v3 based on availability
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com",
    temperature=0.0
)

def simple_min_max_norm(scores: List[float]) -> List[float]:
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s == 0:
        return [1.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]

def get_llm_relevance_score(query: str, content: str) -> float:
    """
    Uses DeepSeek to score the relevance of the content to the query.
    Returns a float between 0 and 100.
    """
    prompt = f"""You are a relevance ranking assistant. 
Query: {query}
Document: {content[:2000]}... (Possible truncation)

Rate the relevance of the document to the query on a scale of 0 to 100.
0 means completely irrelevant, 100 means the exact answer.
Output ONLY the number.
"""
    try:
        response = llm_reranker.invoke(prompt)
        score_str = response.content.strip()
        # Extract number if there's extra text (simple cleanup)
        import re
        match = re.search(r'\d+(\.\d+)?', score_str)
        if match:
            return float(match.group())
        return 0.0
    except Exception as e:
        print(f"  [Reranker Error] {e}")
        return 0.0

def retrieve_and_rerank(query: str, product_filter: str = None) -> str:
    print(f"🔍 Searching for: {query} (Filter: {product_filter})")
    
    # 1. Vectorize Query
    query_vector = embeddings.embed_query(query)
    
    # 2. Retrieve Top 30 Chunks (Hybrid Search)
    filter_dict = {}
    if product_filter:
        filter_dict = {"company_name": product_filter}

    params = {
        "query_embedding": query_vector,
        "query_text": query, 
        "match_count": 30,
        "dense_weight": 0.7, 
        "sparse_weight": 0.3,
        "filter": filter_dict
    }
    
    try:
        results = supabase.rpc("match_parent_chunks_hybrid", params).execute()
        chunks = results.data
    except Exception as e:
        return f"Database Error: {e}"

    if not chunks:
        return "No relevant documents found."

    print(f"  Found {len(chunks)} chunks.")

    # 3. Deduplicate Pages
    # Group by parent_id, keep the max vector score for that page
    unique_pages = {}
    for chunk in chunks:
        pid = chunk['parent_id']
        # dense_score comes from the RPC
        # The RPC returns 'similarity' which is the hybrid weighted score.
        # But for the 0.3 vector / 0.7 LLM calculation later, we arguably want the RETRIEVAL score.
        # Let's use the 'similarity' (hybrid score) as the representative score for the retrieval phase.
        score = chunk['similarity'] 
        
        if pid not in unique_pages:
            unique_pages[pid] = {
                "parent_content": chunk['parent_content'],
                "metadata": chunk['metadata'],
                "retrieval_score": score
            }
        else:
            if score > unique_pages[pid]["retrieval_score"]:
                unique_pages[pid]["retrieval_score"] = score
    
    pages_list = [{"id": pid, **data} for pid, data in unique_pages.items()]
    print(f"  Deduplicated to {len(pages_list)} unique pages.")

    # 4. LLM Reranking (DeepSeek)
    # Weights
    param_retrieval_weight = 0.3
    param_llm_weight = 0.7
    
    retrieval_scores = [p['retrieval_score'] for p in pages_list]
    # Normalize Retrieval Scores locally
    norm_retrieval_scores = simple_min_max_norm(retrieval_scores)
    
    print(f"  🤖 Reranking {len(pages_list)} pages with DeepSeek (Concurrent)...")
    llm_raw_scores = [0.0] * len(pages_list)
    
    import concurrent.futures
    
    # Helper for concurrent execution to preserve order or map back
    def score_page(index, page):
        s = get_llm_relevance_score(query, page['parent_content'])
        return index, s

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {executor.submit(score_page, i, p): i for i, p in enumerate(pages_list)}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, score = future.result()
            llm_raw_scores[idx] = score
            # Optional: simpler logging or just log at end
            # print(f"    [Done] Page {pages_list[idx]['metadata'].get('page_source', '?')} | Score: {score}")

    norm_llm_scores = simple_min_max_norm(llm_raw_scores)

    # 5. Calculate Final Score & Sort
    final_results = []
    for i, page in enumerate(pages_list):
        final_score = (param_retrieval_weight * norm_retrieval_scores[i]) + \
                      (param_llm_weight * norm_llm_scores[i])
        
        final_results.append({
            "content": page['parent_content'],
            "page_num": page['metadata'].get('page_source', '?'),
            "score": final_score
        })
    
    # Sort descending
    final_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Keep Top 10
    top_10 = final_results[:10]
    
    # 6. Merge to String
    output_parts = []
    for i, item in enumerate(top_10):
        header = f"--- Result {i+1} (Page {item['page_num']}, Score: {item['score']:.4f}) ---"
        output_parts.append(f"{header}\n{item['content']}\n")
        
    return "\n".join(output_parts)

if __name__ == "__main__":
    # Test
    q = "ES9的电池续航是多少？标准版和性能版有什么区别？"
    print(retrieve_and_rerank(q))
