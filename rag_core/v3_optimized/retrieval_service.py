import os
import time
import logging
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from supabase import create_client, Client

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

@dataclass
class RetrievalConfig:
    match_count: int = 30
    rrf_k: int = 60
    # Weights for final: Just LLM dominating, but we keep the config clean
    # For Pure LLM Listwise, we can set LLM weight to 1.0 or keep hybrid
    # Let's keep a small Retrieval weight for stability, or 1.0 LLM as per "Optimized" goal
    rerank_retrieval_weight: float = 0.3
    rerank_llm_weight: float = 0.7
    
    # Batch size for Listwise reranking (Optimization)
    ranking_batch_size: int = 10  
    llm_model: str = "deepseek-chat"
    embedding_model: str = "models/text-embedding-004"

class RetrievalServiceV3:
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        
        # 1. Validate Environment
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") 
        self.deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        self.google_api_key = os.environ.get("GOOGLE_API_KEY") 

        if not all([self.supabase_url, self.supabase_key, self.deepseek_key]):
            raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_SERVICE_KEY, or DEEPSEEK_API_KEY")

        # 2. Initialize Clients
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            self.embeddings = GoogleGenerativeAIEmbeddings(model=self.config.embedding_model)
            
            # DeepSeek LLM for Listwise Reranking
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
        """Min-Max normalization."""
        if not scores: return []
        if len(scores) == 1: return [1.0]
        
        min_s = min(scores)
        max_s = max(scores)
        
        if max_s - min_s == 0:
            return [1.0 for _ in scores]
            
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def _get_llm_batch_scores(self, query: str, docs: List[Dict]) -> Dict[int, float]:
        """
        Uses DeepSeek to score a BATCH of documents (Listwise Reranking).
        Significant optimization over Pointwise (V2).
        """
        if not docs:
            return {}

        formatted_docs = ""
        for i, doc in enumerate(docs):
            # Limit content length to avoid massive context
            content_preview = doc['parent_content'][:800].replace("\n", " ") 
            formatted_docs += f"[ID: {i}]\nContent: {content_preview}\n\n"

        prompt = f"""You are a relevance ranking assistant.
I will provide a Query and a list of {len(docs)} Document Snippets (ID: 0 to {len(docs)-1}).
Your task is to rate the relevance of EACH document to the query on a scale of 0 to 100.

Query: "{query}"

Documents:
{formatted_docs}

INSTRUCTIONS:
1. Analyze each document's content against the query.
2. Assign a score (0-100). 0=Irrelevant, 100=Exact Answer.
3. Return a valid JSON object mapping ID (string) to Score (number).
4. No other text.

Example Output:
{{"0": 15, "1": 85, "2": 0}}
"""
        try:
            response = self.llm_reranker.invoke(prompt)
            content = response.content.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON
            scores_map = json.loads(content)
            
            # Convert keys to int and values to float
            final_scores = {}
            for k, v in scores_map.items():
                try:
                    # Map back to original index in this batch
                    final_scores[int(k)] = float(v)
                except:
                    continue
            
            # Fill missing with 0
            for i in range(len(docs)):
                if i not in final_scores:
                    final_scores[i] = 0.0
            
            return final_scores

        except Exception as e:
            logger.warning(f"Batch Reranking failed: {e}")
            # Fallback: return 0s
            return {i: 0.0 for i in range(len(docs))}

    def search(self, query: str, product_filter: str = None) -> str:
        logger.info(f"V3 Optimized Searching for: {query} (Filter: {product_filter}) [Strategy: RRF + LLM Listwise]")
        
        # 1. Vectorize
        try:
            query_vector = self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return f"Error generating embedding: {e}"

        # 2. Retrieve (Hybrid RRF)
        filter_dict = {}
        if product_filter:
            filter_dict = {"company_name": product_filter}

        params = {
            "query_embedding": query_vector,
            "query_text": query, 
            "match_count": self.config.match_count,
            "rrf_k": self.config.rrf_k,
            "filter": filter_dict
        }
        
        try:
            results = self.supabase.rpc("match_parent_chunks_rrf", params).execute()
            chunks = results.data
        except Exception as e:
            logger.error(f"Supabase RPC (RRF) failed: {e}")
            return f"Database Error: {e}"

        if not chunks:
            return "No relevant documents found."

        logger.info(f"Found {len(chunks)} raw chunks via RRF.")

        # 3. Deduplicate Pages
        unique_pages = {}
        for chunk in chunks:
            pid = chunk['parent_id']
            # Map RRF score (kept for reference, low weight)
            score = chunk['rrf_score']
            
            if pid not in unique_pages:
                unique_pages[pid] = {
                    "parent_content": chunk['parent_content'],
                    "metadata": chunk['metadata'],
                    "retrieval_score": score,
                    "page_source": chunk['metadata'].get('page_source', '?')
                }
            else:
                if score > unique_pages[pid]["retrieval_score"]:
                    unique_pages[pid]["retrieval_score"] = score
        
        pages_list = list(unique_pages.values())
        logger.info(f"Deduplicated to {len(pages_list)} unique pages.")

        # 4. LLM Listwise Reranking (DeepSeek)
        logger.info(f"Running LLM Listwise Reranking on {len(pages_list)} pages...")
        llm_raw_scores = [0.0] * len(pages_list)
        
        batches = [pages_list[i:i + self.config.ranking_batch_size] 
                   for i in range(0, len(pages_list), self.config.ranking_batch_size)]
        
        import concurrent.futures
        
        def process_llm_batch(batch_idx, batch_docs):
            start_idx = batch_idx * self.config.ranking_batch_size
            scores_map = self._get_llm_batch_scores(query, batch_docs)
            return start_idx, scores_map

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_batch = {
                executor.submit(process_llm_batch, i, batch): i 
                for i, batch in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    start_global_idx, scores_map = future.result()
                    for local_idx, score in scores_map.items():
                        global_idx = start_global_idx + local_idx
                        if global_idx < len(llm_raw_scores):
                            llm_raw_scores[global_idx] = score
                except Exception as exc:
                    logger.error(f"LLM Batch {batch_idx} exception: {exc}")

        norm_llm_scores = self.normalize_scores(llm_raw_scores)
        
        # 5. Final Score Fusion
        # We don't normalize RRF scores heavily to let LLM dominate
        # But we still use the helper. 
        # Actually, let's normalize RRF too otherwise 0.1 * 0.03 is invisible.
        retrieval_scores = [p['retrieval_score'] for p in pages_list]
        norm_retrieval_scores = self.normalize_scores(retrieval_scores)

        final_results = []
        for i, page in enumerate(pages_list):
            final_score = (self.config.rerank_retrieval_weight * norm_retrieval_scores[i]) + \
                          (self.config.rerank_llm_weight * norm_llm_scores[i])
            
            final_results.append({
                "content": page['parent_content'],
                "page_num": page['page_source'],
                "score": final_score,
                "llm_raw": llm_raw_scores[i]
            })
        
        final_results.sort(key=lambda x: x['score'], reverse=True)
        top_10 = final_results[:10]
        
        # 6. Format Output
        output_parts = []
        for i, item in enumerate(top_10):
            header = f"--- Result {i+1} (Page {item['page_num']}, Score: {item['score']:.4f} [LLM:{item['llm_raw']:.0f}]) ---"
            output_parts.append(f"{header}\n{item['content']}\n")
            
        return "\n".join(output_parts)
    
if __name__ == "__main__":
    try:
        service = RetrievalServiceV3()
        q = "ES9的电池续航是多少？标准版和性能版有什么区别？"
        print(service.search(q))
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    try:
        service = RetrievalServiceV3()
        q = "ES9的电池续航是多少？"
        print(service.search(q))
    except Exception as e:
        print(f"Test failed: {e}")
