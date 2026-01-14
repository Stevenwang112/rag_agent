-- Enable valid extensions
create extension if not exists vector;

-- 1. Create the Parent Documents Table
-- Stores the full content (Markdown + HTML).
create table if not exists document_parents (
  id uuid primary key default gen_random_uuid(),
  content text,                                
  metadata jsonb,                              
  created_at timestamptz default now()
);

-- 2. Create the Child Chunks Table with Full Text Search (FTS) support
-- Dropping table if exists to ensure schema update (WARNING: DELETES DATA)
drop table if exists document_chunks;

create table document_chunks (
  id uuid primary key default gen_random_uuid(),
  parent_id uuid references document_parents(id) on delete cascade,
  content text,
  metadata jsonb,
  embedding vector(768),
  -- Add a generated column for Full Text Search
  -- Using 'simple' config to support mixed language (basic tokenization) better than 'english' for Chinese
  fts tsvector generated always as (to_tsvector('simple', content)) stored,
  created_at timestamptz default now()
);

-- 3. Indexes for Performance
create index on document_chunks using hnsw (embedding vector_cosine_ops);
create index on document_chunks using gin (fts);

-- 4. Enable RLS
alter table document_parents enable row level security;
alter table document_chunks enable row level security;

-- Policies (Check if exists first to avoid errors on re-run)
do $$ 
begin
    if not exists (select from pg_policies where tablename = 'document_parents' and policyname = 'Allow public read access parents') then
        create policy "Allow public read access parents" on document_parents for select using (true);
    end if;

    if not exists (select from pg_policies where tablename = 'document_chunks' and policyname = 'Allow public read access chunks') then
        create policy "Allow public read access chunks" on document_chunks for select using (true);
    end if;

    if not exists (select from pg_policies where tablename = 'document_parents' and policyname = 'Allow public insert access parents') then
        create policy "Allow public insert access parents" on document_parents for insert with check (true);
    end if;

    if not exists (select from pg_policies where tablename = 'document_chunks' and policyname = 'Allow public insert access chunks') then
        create policy "Allow public insert access chunks" on document_chunks for insert with check (true);
    end if;
end $$;

-- 5. Hybrid Search Function (Weighted Sum) - Adapted for Parent Retrieval
-- Based on the logic: dense_weight * VectorScore + sparse_weight * KeywordScore
-- 5. Hybrid Search Function (Weighted Sum) - Adapted for Parent Retrieval
-- Based on the logic: dense_weight * VectorScore + sparse_weight * KeywordScore
-- UPDATED: Now performs a JOIN to return the FULL Parent Page Content directly.

-- Drop the function first to allow return type changes
DROP FUNCTION IF EXISTS match_parent_chunks_hybrid(vector, text, float, float, int, jsonb);

create or replace function match_parent_chunks_hybrid (
  query_embedding vector(768),
  query_text text,
  dense_weight float default 0.7,
  sparse_weight float default 0.3,
  match_count int default 10,
  filter jsonb default '{}'
) returns table (
  id uuid,
  parent_id uuid,
  chunk_content text,           -- The small chunk used for matching
  parent_content text,          -- The FULL page content (Markdown + HTML)
  metadata jsonb,
  dense_score float,
  sparse_score real,
  similarity float              -- The final weighted score
) language plpgsql stable as $$
begin
  return query
  select
    dc.id,
    dc.parent_id,
    dc.content as chunk_content,
    dp.content as parent_content, -- JOINing to get the full parent content
    dc.metadata,
    (1 - (dc.embedding <=> query_embedding)) as dense_score,
    ts_rank_cd(dc.fts, websearch_to_tsquery('simple', query_text)) as sparse_score,
    
    -- Final Score Calculation
    (dense_weight * (1 - (dc.embedding <=> query_embedding))) + 
    (sparse_weight * ts_rank_cd(dc.fts, websearch_to_tsquery('simple', query_text))) as similarity
    
  from document_chunks dc
  join document_parents dp on dc.parent_id = dp.id
  where dc.metadata @> filter
  order by similarity desc
  limit match_count;
end;
$$;
