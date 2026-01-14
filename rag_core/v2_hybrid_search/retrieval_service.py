import os
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from supabase import create_client, Client

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env vars once at module level, but validate in class
load_dotenv()

@dataclass
class RetrievalConfig:
    match_count: int = 30
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rerank_retrieval_weight: float = 0.3
    rerank_llm_weight: float = 0.7
    max_rerank_workers: int = 10
    llm_model: str = "deepseek-chat"
    embedding_model: str = "models/text-embedding-004"

class RetrievalService:
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        
        # 1. Validate Environment
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") 
        self.deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        # Implicit for Google Embeddings, but good to check if explicity needed usually validation logic differs
        self.google_api_key = os.environ.get("GOOGLE_API_KEY") 

        if not all([self.supabase_url, self.supabase_key, self.deepseek_key]):
            raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_SERVICE_KEY, or DEEPSEEK_API_KEY")

        # 2. Initialize Clients
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            self.embeddings = GoogleGenerativeAIEmbeddings(model=self.config.embedding_model)
            self.llm_reranker = ChatOpenAI(
                model=self.config.llm_model,
                openai_api_key=self.deepseek_key,
                openai_api_base="https://api.deepseek.com",
                temperature=0.0
            )
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise

    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        """
        Min-Max normalization.
        """
        if not scores:
            return []
        if len(scores) == 1:
            return [1.0]
        
        min_s = min(scores)
        max_s = max(scores)
        
        if max_s - min_s == 0:
            return [1.0 for _ in scores]
            
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def _get_llm_relevance_score(self, query: str, content: str) -> float:
        """
        Uses DeepSeek to score relevance (0-100).
        """
        prompt = f"""You are a relevance ranking assistant. 
Query: {query}
Document: {content[:2000]}...
Rate the relevance of the document to the query on a scale of 0 to 100.
0 means irrelevant, 100 means exact answer.
Output ONLY the number.
"""
        try:
            response = self.llm_reranker.invoke(prompt)
            score_str = response.content.strip()
            import re
            match = re.search(r'\d+(\.\d+)?', score_str)
            if match:
                return float(match.group())
            return 0.0
        except Exception as e:
            logger.warning(f"Reranker LLM call failed: {e}")
            return 0.0

    def search(self, query: str, product_filter: str = None) -> str:
        logger.info(f"Searching for: {query} (Filter: {product_filter})")
        
        # 1. Vectorize
        try:
            query_vector = self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return f"Error generating embedding: {e}"

        # 2. Retrieve (Hybrid)
        filter_dict = {}
        if product_filter:
            filter_dict = {"company_name": product_filter}

        params = {
            "query_embedding": query_vector,
            "query_text": query, 
            "match_count": self.config.match_count,
            "dense_weight": self.config.dense_weight, 
            "sparse_weight": self.config.sparse_weight,
            "filter": filter_dict
        }
        
        try:
            results = self.supabase.rpc("match_parent_chunks_hybrid", params).execute()
            chunks = results.data
        except Exception as e:
            logger.error(f"Supabase RPC failed: {e}")
            return f"Database Error: {e}"

        if not chunks:
            return "No relevant documents found."

        logger.info(f"Found {len(chunks)} raw chunks.")

        # 3. Deduplicate Pages
        unique_pages = {}
        for chunk in chunks:
            pid = chunk['parent_id']
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
        logger.info(f"Deduplicated to {len(pages_list)} unique pages.")

        # 4. LLM Reranking
        retrieval_scores = [p['retrieval_score'] for p in pages_list]
        norm_retrieval_scores = self.normalize_scores(retrieval_scores)
        
        logger.info(f"Reranking {len(pages_list)} pages with DeepSeek...")
        llm_raw_scores = [0.0] * len(pages_list)
        
        import concurrent.futures
        
        def score_task(index, page):
            s = self._get_llm_relevance_score(query, page['parent_content'])
            return index, s

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_rerank_workers) as executor:
            future_to_idx = {executor.submit(score_task, i, p): i for i, p in enumerate(pages_list)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx, score = future.result()
                llm_raw_scores[idx] = score

        norm_llm_scores = self.normalize_scores(llm_raw_scores)

        # 5. Final Score
        final_results = []
        for i, page in enumerate(pages_list):
            final_score = (self.config.rerank_retrieval_weight * norm_retrieval_scores[i]) + \
                          (self.config.rerank_llm_weight * norm_llm_scores[i])
            
            final_results.append({
                "content": page['parent_content'],
                "page_num": page['metadata'].get('page_source', '?'),
                "score": final_score
            })
        
        final_results.sort(key=lambda x: x['score'], reverse=True)
        top_10 = final_results[:10]
        
        # 6. Format Output
        output_parts = []
        for i, item in enumerate(top_10):
            header = f"--- Result {i+1} (Page {item['page_num']}, Score: {item['score']:.4f}) ---"
            output_parts.append(f"{header}\n{item['content']}\n")
            
        return "\n".join(output_parts)

if __name__ == "__main__":
    # Test V3
    try:
        service = RetrievalService()
        q = "ES9的电池续航是多少？标准版和性能版有什么区别？"
        print(service.search(q))
    except Exception as e:
        print(f"Test failed: {e}")
