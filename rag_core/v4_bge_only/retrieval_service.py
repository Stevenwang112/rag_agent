import os
import time
import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from supabase import create_client, Client
from sentence_transformers import CrossEncoder

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

@dataclass
class RetrievalConfig:
    match_count: int = 30
    rrf_k: int = 60
    # Weights: BGE dominance
    rerank_retrieval_weight: float = 0.3
    rerank_bge_weight: float = 0.7
    
    embedding_model: str = "models/text-embedding-004"
    bge_model_name: str = "BAAI/bge-reranker-v2-m3"

class RetrievalServiceV4:
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        
        # 1. Validate Environment
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") 
        self.google_api_key = os.environ.get("GOOGLE_API_KEY") 

        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_SERVICE_KEY")

        # 2. Initialize Clients
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            self.embeddings = GoogleGenerativeAIEmbeddings(model=self.config.embedding_model)
            
            # BGE Reranker (Standard Loading)
            logger.info(f"Loading BGE Reranker V4: {self.config.bge_model_name}...")
            self.bge_reranker = CrossEncoder(self.config.bge_model_name, automodel_args={"torch_dtype": "auto"})
            logger.info("BGE Reranker loaded successfully.")
            
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
        
        # Edge case: All scores are 0.0 (failed rerank)
        if max_s == 0:
            return [0.0 for _ in scores]
        
        if max_s - min_s == 0:
            # All scores are identical and non-zero => treat as equal max relevance
            return [1.0 for _ in scores]
            
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def _get_bge_scores(self, query: str, docs: List[Dict]) -> List[float]:
        """
        Uses BGE-M3 CrossEncoder to score documents.
        """
        if not docs:
            return []
        
        # Prepare pairs: (query, doc_content)
        # Limit content to 2000 chars to enable reasonable speed
        pairs = [(query, doc['parent_content'][:2000]) for doc in docs]
        
        try:
            # CrossEncoder returns list of floats (logits)
            scores = self.bge_reranker.predict(pairs)
            return scores.tolist()
        except Exception as e:
            logger.error(f"BGE prediction failed: {e}")
            return [0.0] * len(docs)

    def search(self, query: str, product_filter: str = None) -> str:
        logger.info(f"V4 (BGE) Searching for: {query} (Filter: {product_filter}) [Strategy: RRF + BGE Rerank]")
        
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
            
        # 3. Deduplicate Pages
        unique_pages = {}
        for chunk in chunks:
            pid = chunk['parent_id']
            # We keep RRF score just for reference (or small weight)
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

        # 4. BGE Reranking
        logger.info(f"Running BGE Reranking on {len(pages_list)} pages...")
        bge_raw_scores = self._get_bge_scores(query, pages_list)
        norm_bge_scores = self.normalize_scores(bge_raw_scores)
        
        # Also normalize retrieval scores
        retrieval_scores = [p['retrieval_score'] for p in pages_list]
        norm_retrieval_scores = self.normalize_scores(retrieval_scores)

        # 5. Final Score Fusion
        final_results = []
        for i, page in enumerate(pages_list):
            final_score = (self.config.rerank_retrieval_weight * norm_retrieval_scores[i]) + \
                          (self.config.rerank_bge_weight * norm_bge_scores[i])
            
            final_results.append({
                "content": page['parent_content'],
                "page_num": page['page_source'],
                "score": final_score,
                "bge_raw": bge_raw_scores[i]
            })
        
        final_results.sort(key=lambda x: x['score'], reverse=True)
        top_10 = final_results[:10]
        
        # 6. Format Output
        output_parts = []
        for i, item in enumerate(top_10):
            header = f"--- Result {i+1} (Page {item['page_num']}, Score: {item['score']:.4f} [BGE:{item['bge_raw']:.2f}]) ---"
            output_parts.append(f"{header}\n{item['content']}\n")
            
        return "\n".join(output_parts)
    
if __name__ == "__main__":
    try:
        service = RetrievalServiceV4()
        q = "ES9的电池续航是多少？"
        print(service.search(q))
    except Exception as e:
        print(f"Test failed: {e}")
