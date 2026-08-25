import os
import sys
import json
import asyncio
import io
from xml.etree import ElementTree
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
from pathlib import Path
from dotenv import load_dotenv
import httpx
from pypdf import PdfReader
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

if not openai_client.api_key or not supabase_url or not supabase_key:
    raise ValueError("Missing required environment variables (OpenAI or Supabase)")

supabase: Client = create_client(supabase_url, supabase_key)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 3000, chunk_overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        current_chunk = text[start:min(end, text_length)]
        if end < text_length:
            overlap_start_index = max(start, end - chunk_overlap)
            potential_break_area = text[overlap_start_index:end]
            para_break = potential_break_area.rfind('\n\n')
            if para_break != -1:
                end = overlap_start_index + para_break + 2
                current_chunk = text[start:end]
            elif '. ' in potential_break_area:
                 sentence_break = potential_break_area.rfind('. ')
                 if sentence_break != -1:
                      end = overlap_start_index + sentence_break + 2
                      current_chunk = text[start:end]
            else:
                 end = start + chunk_size
        if len(current_chunk.strip()) > chunk_overlap / 2:
            chunks.append(current_chunk.strip())
        start = max(start + 1, end - chunk_overlap)
        if start <= (len(chunks) - 1) * (chunk_size - chunk_overlap) if chunks else 0:
            start = end
    return [c for c in chunks if c]

async def get_title_and_summary(chunk: str, url: str, context: str = "page content") -> Dict[str, str]:
    system_prompt = f"""You are an AI that extracts titles and summaries from {context} chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its main title. If it's a middle chunk, derive a descriptive title reflecting the chunk's content. Use the URL path as a hint if needed.
    For the summary: Create a concise summary (1-2 sentences) of the main points in this specific chunk.
    Keep both title and summary concise but informative."""
    try:
        content_preview = chunk[:1500] + "..." if len(chunk) > 1500 else chunk
        url_path = urlparse(url).path
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL Path: {url_path}\n\nContent Chunk:\n{content_preview}"}
            ],
            response_format={ "type": "json_object" }
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "title": result.get("title", f"Chunk from {url_path}"),
            "summary": result.get("summary", "Summary unavailable")
        }
    except Exception as e:
        print(f"Error getting title/summary for {url}: {e}")
        return {"title": f"Content from {urlparse(url).path}", "summary": "Error generating summary."}

async def get_embedding(text: str, url: str) -> List[float]:
    if not text: return [0.0] * 1536
    try:
        text_to_embed = text.replace("\n", " ").strip()
        if not text_to_embed:
             return [0.0] * 1536
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[text_to_embed]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for {url}: {e}")
        return [0.0] * 1536

async def process_chunk(chunk_content: str, chunk_number: int, url: str, source_tag: str, is_pdf: bool = False, pdf_page_num: Optional[int] = None) -> Optional[ProcessedChunk]:
    if not chunk_content.strip():
        return None
    context_type = "PDF page" if is_pdf else "web page"
    extracted = await get_title_and_summary(chunk_content, url, context=context_type)
    embedding = await get_embedding(chunk_content, url)
    metadata = {
        "source": source_tag,
        "chunk_size": len(chunk_content),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path,
        "is_pdf": is_pdf,
    }
    if is_pdf and pdf_page_num is not None:
        metadata["pdf_page_number"] = pdf_page_num
    if is_pdf and extracted['title'].startswith("Chunk from") and pdf_page_num == 1:
         extracted['title'] = f"PDF Document: {metadata['url_path']}"
    elif is_pdf and extracted['title'].startswith("Chunk from"):
         extracted['title'] = f"PDF: {metadata['url_path']} (Page approx {pdf_page_num})"
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk_content,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk, table_name: str):
    if not chunk or not chunk.content: return None
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        result = await asyncio.to_thread(supabase.table(table_name).insert(data).execute)
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url} into table '{table_name}'")
        return result
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e):
             print(f"Skipping duplicate chunk {chunk.chunk_number} for {chunk.url} in table '{table_name}'")
        else:
            print(f"Error inserting chunk {chunk.chunk_number} for {chunk.url} into table '{table_name}': {e}")
        return None

async def process_and_store_text(full_text: str, url: str, source_tag: str, table_name: str, is_pdf: bool = False):
    chunks = chunk_text(full_text)
    print(f"Processing {len(chunks)} chunks for {url} (PDF: {is_pdf})")
    tasks = []
    for i, chunk_content in enumerate(chunks):
        pdf_page = 1 if is_pdf and i == 0 else None
        tasks.append(process_chunk(chunk_content, i, url, source_tag, is_pdf=is_pdf, pdf_page_num=pdf_page))
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [
        insert_chunk(chunk, table_name)
        for chunk in processed_chunks if chunk
    ]
    valid_insert_tasks = [task for task in insert_tasks if task is not None]
    if valid_insert_tasks:
        results = await asyncio.gather(*valid_insert_tasks)
        print(f"Finished storing {len(results)} chunks for {url}")
    else:
         print(f"No valid chunks to insert for {url}")

async def parse_pdf_from_url(url: str) -> Optional[str]:
    print(f"Attempting to parse PDF: {url}")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        if 'application/pdf' not in response.headers.get('content-type', '').lower():
            print(f"Warning: URL {url} did not return content-type 'application/pdf'. Skipping PDF parse.")
            return None
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += f"\n\n--- PDF Page {page_num + 1} ---\n\n" + page_text.strip()
            except Exception as page_error:
                print(f"Error extracting text from page {page_num + 1} of {url}: {page_error}")
        print(f"Successfully extracted text from PDF: {url} (Length: {len(full_text)})")
        return full_text.strip()
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
        return None
    except httpx.HTTPStatusError as exc:
        print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
        return None
    except Exception as e:
        print(f"Failed to parse PDF {url}: {e}")
        return None

async def crawl_parallel(urls: List[str], source_tag: str, table_name: str, max_concurrent: int = 3):
    print(f"Starting crawl for source '{source_tag}' -> table '{table_name}'")
    browser_config = BrowserConfig(headless=True, verbose=False, extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"])
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    semaphore = asyncio.Semaphore(max_concurrent)
    async def process_url(url: str):
        async with semaphore:
            await asyncio.sleep(0.5)
            print(f"Processing URL: {url}")
            is_pdf = url.lower().endswith('.pdf') or ".pdf?" in url.lower()
            if is_pdf:
                pdf_text = await parse_pdf_from_url(url)
                if pdf_text:
                    await process_and_store_text(pdf_text, url, source_tag, table_name, is_pdf=True)
                else:
                    print(f"Could not process PDF text for {url}")
            else:
                try:
                    result = await crawler.arun(url=url, config=crawl_config, session_id=f"session_{source_tag}")
                    if result.success and result.markdown_v2 and result.markdown_v2.raw_markdown:
                        print(f"Crawled HTML: {url} (Source: {source_tag})")
                        await process_and_store_text(result.markdown_v2.raw_markdown, url, source_tag, table_name, is_pdf=False)
                    elif result.success:
                        print(f"Crawled HTML but no markdown content found: {url}")
                    else:
                        print(f"Failed HTML crawl: {url} - Error: {result.error_message}")
                except Exception as crawl_err:
                     print(f"Exception during HTML crawl for {url}: {crawl_err}")
    tasks = [process_url(url) for url in urls]
    await asyncio.gather(*tasks)
    await crawler.close()
    print(f"Finished crawl for source '{source_tag}' -> table '{table_name}'")

def get_urls_from_config(source_config: Dict) -> List[str]:
    urls = []
    sitemap_path = source_config.get("sitemap_path")
    direct_urls = source_config.get("direct_urls", [])
    if sitemap_path:
        try:
            sitemap_file = Path(sitemap_path)
            if not sitemap_file.is_file():
                 print(f"Error: Sitemap file not found at {sitemap_path}")
            else:
                tree = ElementTree.parse(sitemap_file)
                root = tree.getroot()
                namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                sitemap_urls = [loc.text for loc in root.findall('.//ns:url/ns:loc', namespaces)]
                if not sitemap_urls:
                    sitemap_urls = [loc.text for loc in root.findall('{http://www.sitemaps.org/schemas/sitemap/0.9}url/{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
                print(f"Found {len(sitemap_urls)} URLs in {sitemap_path}")
                urls.extend(sitemap_urls)
        except Exception as e:
            print(f"Error reading sitemap {sitemap_path}: {e}")
    if direct_urls:
        print(f"Adding {len(direct_urls)} direct URLs.")
        urls.extend(direct_urls)
    valid_urls = []
    for u in urls:
        if u and isinstance(u, str) and (u.startswith('http://') or u.startswith('https://')):
             valid_urls.append(u)
        else:
             print(f"Skipping invalid URL: {u}")
    return list(set(valid_urls))

async def main():
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)
    sources = {
        "ovgu": {
            "sitemap_path": data_dir / "ovgu.de_sitemap.xml",
            "direct_urls": [],
            "source_tag": "ovgu_docs",
            "table_name": "ovgu_pages"
        },
        "magdeburg": {
            "sitemap_path": data_dir / "magdeburg.de_sitemap.xml",
            "direct_urls": [],
            "source_tag": "magdeburg_general_docs",
            "table_name": "magdeburg_pages"
        },
        "fin": {
            "sitemap_path": data_dir / "fin.de_sitemap.xml",
            "direct_urls": [],
            "source_tag": "fin_docs",
            "table_name": "fin_pages"
        }
    }
    max_concurrent_crawls = 2
    for source_key, config in sources.items():
         sitemap_path = config.get("sitemap_path")
         if sitemap_path:
             Path(sitemap_path).touch(exist_ok=True)
    print(f"Place sitemaps (if applicable) in: {data_dir.resolve()}")
    for source_key, config in sources.items():
        print(f"\n--- Starting {source_key.upper()} Ingestion ---")
        urls_to_crawl = get_urls_from_config(config)
        if urls_to_crawl:
            await crawl_parallel(
                urls_to_crawl,
                config["source_tag"],
                config["table_name"],
                max_concurrent_crawls
            )
        else:
            print(f"No URLs found to process for {source_key.upper()}.")
    print("\n--- Ingestion Complete ---")

if __name__ == "__main__":
    asyncio.run(main())