-- supabase_schema_separate.sql

-- Drop existing objects first to ensure a clean setup (deletes existing data!)
DROP FUNCTION IF EXISTS public.match_ovgu_pages(vector, integer);
DROP FUNCTION IF EXISTS public.match_magdeburg_pages(vector, integer);
DROP FUNCTION IF EXISTS public.match_fin_pages(vector, integer);
DROP TABLE IF EXISTS public.ovgu_pages CASCADE;
DROP TABLE IF EXISTS public.magdeburg_pages CASCADE;
DROP TABLE IF EXISTS public.fin_pages CASCADE;

-- Enable the pgvector extension
create extension if not exists vector with schema public;

-- === OVGU Table ===

-- Create the OVGU documentation chunks table
create table public.ovgu_pages (
    id bigserial primary key,
    url text not null,
    chunk_number integer not null,
    title text not null,
    summary text not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536), -- OpenAI text-embedding-3-small dimension
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(url, chunk_number)
);

-- Add comments to explain the columns
comment on table public.ovgu_pages is 'Stores chunks of text content scraped from the OVGU website.';
comment on column public.ovgu_pages.url is 'The original URL from which the content chunk was derived.';
comment on column public.ovgu_pages.chunk_number is 'The sequential index of the chunk within the original page content.';
comment on column public.ovgu_pages.title is 'A generated or extracted title for the content chunk.';
comment on column public.ovgu_pages.summary is 'A short summary of the content chunk.';
comment on column public.ovgu_pages.content is 'The actual text content of the chunk.';
comment on column public.ovgu_pages.metadata is 'Additional metadata about the chunk (e.g., source tag, crawl time, page number for PDFs).';
comment on column public.ovgu_pages.embedding is 'Vector embedding of the content for semantic search.';
comment on column public.ovgu_pages.created_at is 'Timestamp when the record was created.';

-- Create an index for better vector similarity search performance on OVGU table
create index if not exists ovgu_pages_embedding_ivfflat_idx on public.ovgu_pages using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Create an index on metadata for faster filtering on OVGU table
create index if not exists idx_ovgu_pages_metadata on public.ovgu_pages using gin (metadata jsonb_path_ops);

-- Create a function to search for OVGU documentation chunks
create or replace function public.match_ovgu_pages (
  query_embedding vector(1536),
  match_count int default 3
) returns table (
  id bigint,
  url text,
  chunk_number integer,
  title text,
  summary text,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    ovgu_pages.id,
    ovgu_pages.url,
    ovgu_pages.chunk_number,
    ovgu_pages.title,
    ovgu_pages.summary,
    ovgu_pages.content,
    ovgu_pages.metadata,
    1 - (ovgu_pages.embedding <=> query_embedding) as similarity
  from public.ovgu_pages
  order by ovgu_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- === Magdeburg General Table ===

-- Create the Magdeburg general documentation chunks table
create table public.magdeburg_pages (
    id bigserial primary key,
    url text not null,
    chunk_number integer not null,
    title text not null,
    summary text not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(url, chunk_number)
);

comment on table public.magdeburg_pages is 'Stores chunks of text content scraped from general Magdeburg websites/sources.';

-- Create an index for better vector similarity search performance on Magdeburg table
create index if not exists magdeburg_pages_embedding_ivfflat_idx on public.magdeburg_pages using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Create an index on metadata for faster filtering on Magdeburg table
create index if not exists idx_magdeburg_pages_metadata on public.magdeburg_pages using gin (metadata jsonb_path_ops);

-- Create a function to search for Magdeburg documentation chunks
create or replace function public.match_magdeburg_pages (
  query_embedding vector(1536),
  match_count int default 3
) returns table (
  id bigint, url text, chunk_number integer, title text, summary text, content text, metadata jsonb, similarity float
)
language plpgsql
as $$
begin
  return query
  select
    magdeburg_pages.id,
    magdeburg_pages.url,
    magdeburg_pages.chunk_number,
    magdeburg_pages.title,
    magdeburg_pages.summary,
    magdeburg_pages.content,
    magdeburg_pages.metadata,
    1 - (magdeburg_pages.embedding <=> query_embedding) as similarity
  from public.magdeburg_pages
  order by magdeburg_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- === FIN Table ===

-- Create the FIN documentation chunks table
create table public.fin_pages (
    id bigserial primary key,
    url text not null,
    chunk_number integer not null,
    title text not null,
    summary text not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(url, chunk_number)
);

comment on table public.fin_pages is 'Stores chunks of text content scraped from the FIN faculty website and documents.';

-- Create an index for better vector similarity search performance on FIN table
create index if not exists fin_pages_embedding_ivfflat_idx on public.fin_pages using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Create an index on metadata for faster filtering on FIN table
create index if not exists idx_fin_pages_metadata on public.fin_pages using gin (metadata jsonb_path_ops);

-- Create a function to search for FIN documentation chunks
create or replace function public.match_fin_pages (
  query_embedding vector(1536),
  match_count int default 3
) returns table (
  id bigint, url text, chunk_number integer, title text, summary text, content text, metadata jsonb, similarity float
)
language plpgsql
as $$
begin
  return query
  select
    fin_pages.id,
    fin_pages.url,
    fin_pages.chunk_number,
    fin_pages.title,
    fin_pages.summary,
    fin_pages.content,
    fin_pages.metadata,
    1 - (fin_pages.embedding <=> query_embedding) as similarity
  from public.fin_pages
  order by fin_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- === RLS Policies (Apply to ALL THREE tables) ===
-- Grant usage on schema public to anon and authenticated if needed for non-service_role keys
-- grant usage on schema public to anon, authenticated;
-- grant execute on function public.match_ovgu_pages(vector, integer) to anon, authenticated;
-- grant execute on function public.match_magdeburg_pages(vector, integer) to anon, authenticated;
-- grant execute on function public.match_fin_pages(vector, integer) to anon, authenticated;

-- OVGU
alter table public.ovgu_pages enable row level security;
create policy "Allow public read access on ovgu" on public.ovgu_pages for select to public using (true);

-- Magdeburg General
alter table public.magdeburg_pages enable row level security;
create policy "Allow public read access on magdeburg" on public.magdeburg_pages for select to public using (true);

-- FIN
alter table public.fin_pages enable row level security;
create policy "Allow public read access on fin" on public.fin_pages for select to public using (true);