-- RRF (Reciprocal Rank Fusion) Hybrid Search
-- implementation for Dense (Vector) + Sparse (BM25/FTS)

-- Function: match_parent_chunks_rrf
-- Arguments:
--   query_embedding: The vector embedding of the user query
--   query_text: The raw text of the user query (for FTS)
--   match_count: The number of candidates to retrieve from EACH method before fusing (and final limit)
--   rrf_k: The 'k' constant in RRF formula (default 60).
--          Score = 1/(k + rank_dense) + 1/(k + rank_sparse)
--   filter: JSONB filter for metadata (e.g. {"company_name": "ES9"})

CREATE OR REPLACE FUNCTION match_parent_chunks_rrf (
  query_embedding vector(768),
  query_text text,
  match_count int default 30,
  rrf_k int default 60,
  filter jsonb default '{}'
) returns table (
  id uuid,
  parent_id uuid,
  chunk_content text,
  parent_content text,
  metadata jsonb,
  rrf_score double precision
) language plpgsql stable as $$
begin
  return query
  WITH 
  -- 1. Get Top N Document Chunks via Dense Vector Search
  dense_results AS (
      SELECT 
          dc.id,
          -- Use ROW_NUMBER since we want unique ranks 1..N
          ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_embedding) as dense_rank
      FROM document_chunks dc
      WHERE dc.metadata @> filter
      ORDER BY dc.embedding <=> query_embedding
      LIMIT match_count
  ),
  -- 2. Get Top N Document Chunks via Sparse Keyword Search (FTS)
  sparse_results AS (
      SELECT 
          dc.id,
          ROW_NUMBER() OVER (ORDER BY ts_rank_cd(dc.fts, websearch_to_tsquery('simple', query_text)) DESC) as sparse_rank
      FROM document_chunks dc
      WHERE dc.metadata @> filter
        -- Only include if it actually matches keywords
        AND dc.fts @@ websearch_to_tsquery('simple', query_text)
      ORDER BY ts_rank_cd(dc.fts, websearch_to_tsquery('simple', query_text)) DESC
      LIMIT match_count
  ),
  -- 3. Fuse Results using RRF Formula
  rrf_scores AS (
      SELECT
          COALESCE(d.id, s.id) as chunk_id,
          (
            COALESCE(1.0 / (rrf_k + d.dense_rank), 0.0) + 
            COALESCE(1.0 / (rrf_k + s.sparse_rank), 0.0)
          )::double precision as score
      FROM dense_results d
      FULL OUTER JOIN sparse_results s ON d.id = s.id
  )
  -- 4. Join back to get content and return Top Results
  SELECT
      dc.id,
      dc.parent_id,
      dc.content as chunk_content,
      dp.content as parent_content,
      dc.metadata,
      rs.score as rrf_score
  FROM rrf_scores rs
  JOIN document_chunks dc ON rs.chunk_id = dc.id
  JOIN document_parents dp ON dc.parent_id = dp.id
  ORDER BY rs.score DESC
  LIMIT match_count;
end;
$$;
