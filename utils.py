# utils.py
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()

def get_model():
    llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAIModel(llm, base_url=base_url, api_key=api_key)

def get_openai_client() -> AsyncOpenAI:
     api_key = os.getenv('OPENAI_API_KEY')
     if not api_key:
         raise ValueError("OPENAI_API_KEY environment variable not set.")
     return AsyncOpenAI(api_key=api_key)

def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_KEY environment variable not set.")
    return create_client(url, key)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    if not text or not isinstance(text, str):
         print("Warning: Invalid text provided for embedding. Returning zero vector.")
         return [0.0] * 1536
    try:
        text_to_embed = text.replace("\n", " ").strip()
        if not text_to_embed: # Handle empty string after stripping
             return [0.0] * 1536
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[text_to_embed]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text '{text[:50]}...': {e}")
        return [0.0] * 1536