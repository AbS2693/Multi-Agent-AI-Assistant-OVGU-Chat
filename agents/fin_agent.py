# agents/fin_agent.py
from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import os
import asyncio

from pydantic_ai import Agent, ModelRetry, RunContext
from supabase import Client
from openai import AsyncOpenAI
from typing import List

from utils import get_model, get_embedding, get_supabase_client, get_openai_client

load_dotenv()
logfire.configure(send_to_logfire='if-token-present')

model = get_model()

@dataclass
class FinAgentDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt_fin = """
You are an expert assistant specializing in questions about the Faculty of Informatics (FIN) at OVGU.
Your primary goal is to provide accurate, comprehensive, and well-sourced answers based *exclusively* on the information retrieved by the 'retrieve_fin_documentation' tool.

**Core Instructions:**
1.  **Use the Tool First**: Always use the `retrieve_fin_documentation` tool to gather relevant context before formulating your answer.
2.  **Synthesize Information**: Do not just copy-paste chunks. Read through *all* the provided context snippets. Synthesize the relevant information from these snippets into a coherent and comprehensive answer.
3.  **Strictly Grounded**: Base your answer *only* on the information present in the retrieved context. Do not add external knowledge, assumptions, or information not explicitly found in the provided sources.
4.  **Cite Sources**: When possible, mention the source URL provided in the context for the information you are presenting. You can add citations like `(Source: [URL])`.
5.  **Handle Missing Information**: If the retrieved context does *not* contain the answer to the user's query, explicitly state that. For example: "Based on the provided FIN documentation, I could not find specific information about [topic]." Do not attempt to guess or provide related but irrelevant information.
6.  **Formatting**: Format your response clearly using markdown (bolding, bullet points) for readability.

**Specific Topic Guidelines:**
*   For location questions: Include building name/number (G29 etc.), room number, and map links if available in the context.
*   For people-related questions: Include full name, title, research group, contact info, and profile links if available in the context.
*   For study-related questions: Include program name, requirements, deadlines, and relevant links if available in the context.

Do not answer general OVGU questions (library, mensa, general admission) unless they specifically relate to FIN procedures. Do not answer questions about other faculties or Magdeburg city topics.
"""

fin_expert_agent = Agent(
    model,
    system_prompt=system_prompt_fin,
    deps_type=FinAgentDeps,
    retries=1
)

@fin_expert_agent.tool
async def retrieve_fin_documentation(ctx: RunContext[FinAgentDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks about the Faculty of Informatics (FIN) at OVGU based on the user query using RAG.

    Args:
        ctx: The context including Supabase and OpenAI clients.
        user_query: The user's question about FIN.

    Returns:
        A formatted string containing the top 7 most relevant documentation chunks about FIN, including source URLs.
    """
    print(f"FIN Agent: Retrieving docs for query: {user_query}")
    try:
        if not hasattr(ctx.deps, 'openai_client') or not ctx.deps.openai_client:
             print("FIN Agent Error: OpenAI client missing.")
             return "Error: Agent configuration issue (OpenAI client missing)."
        if not hasattr(ctx.deps, 'supabase') or not ctx.deps.supabase:
             print("FIN Agent Error: Supabase client missing.")
             return "Error: Agent configuration issue (Supabase client missing)."

        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        result = await asyncio.to_thread(
            ctx.deps.supabase.rpc(
                'match_fin_pages',
                {
                    'query_embedding': query_embedding,
                    'match_count': 7,  # Increased from 3 to 7 for better coverage
                }
            ).execute
        )

        if not result.data:
            print("FIN Agent: No relevant documentation found.")
            return "I could not find specific information about that topic in the FIN documentation."

        # Format results with better structure
        formatted_chunks = []
        for doc in result.data:
            # Extract metadata
            url = doc.get('url', 'N/A')
            content = doc.get('content', 'N/A')
            metadata = doc.get('metadata', {})
            
            # Format page info if available
            page_info = ""
            if metadata.get('is_pdf') and 'pdf_page_number' in metadata:
                page_info = f" (Page {metadata['pdf_page_number']})"
            
            # Format the chunk with clear sections
            chunk_text = f"""
**Source**: {url}{page_info}
**Content**:
{content}
"""
            formatted_chunks.append(chunk_text)

        print(f"FIN Agent: Found {len(formatted_chunks)} relevant chunks.")
        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"FIN Agent: Error retrieving documentation: {e}")
        return "An error occurred while retrieving FIN documentation."