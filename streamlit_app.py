from langgraph.errors import GraphRecursionError
from datetime import datetime
import streamlit as st
import asyncio
import uuid
import os
import sys
import traceback
import re

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from graph.agent_graph import university_agent_graph, UniversityAgentState
except ImportError as e:
    st.error(f"Could not import the agent graph: {str(e)}")
    st.error(f"Python path: {sys.path}")
    st.error(f"Current directory: {os.getcwd()}")
    st.stop()

st.set_page_config(
    page_title="OVGU / FIN / Magdeburg Assistant",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e6f3ff;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "last_query" not in st.session_state:
    st.session_state.last_query = None

def add_message_to_history(role: str, content: str):
    """Add a message to the chat history with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })

def format_response(response: str) -> str:
    """Format the response to make it more readable and handle source links."""
    # Ensure proper line breaks in markdown
    formatted = response.replace("\n", "  \n")
    
    # Make source links clickable (simple regex approach)
    # Pattern to find (Source: URL) or [Source: URL] or Source: URL
    pattern = r'(\(|\[)?Source:\s*(https?://[\w\./\-\_\?=&%]+)(\)|\])?'
    
    def replace_link(match):
        url = match.group(2)
        return f'([Source]({url}))'
        
    formatted = re.sub(pattern, replace_link, formatted, flags=re.IGNORECASE)
    
    return formatted

st.title("🎓 OVGU / FIN / Magdeburg Assistant")
st.caption("Ask about OVGU (General), FIN Faculty, or Magdeburg City!")

with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.markdown("---")
    st.markdown(f"Session ID: `{st.session_state.session_id}`")
    st.markdown("### Chat History")
    for msg in reversed(st.session_state.chat_history):
        st.markdown(f"**{msg['role'].title()}** ({msg['timestamp']})")
        st.markdown(msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content'])

# Display chat history with better formatting
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar="🎓" if message["role"] == "assistant" else "👤"):
        st.markdown(format_response(message["content"]))

# Get user input
user_input = st.chat_input("Ask your question...", disabled=st.session_state.processing)

async def process_user_input(query: str):
    """Process user input and get agent response."""
    try:
        # Check if this is a duplicate query
        if query == st.session_state.last_query:
            st.warning("You just asked the same question. Please try rephrasing or ask a different question.")
            return
        
        st.session_state.processing = True
        st.session_state.last_query = query
        add_message_to_history("user", query)
        
        # Show thinking message
        with st.chat_message("assistant", avatar="🎓"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Thinking...")
            
            # Get agent response
            config = {"configurable": {"thread_id": st.session_state.session_id}}
            initial_state = {
                "user_query": query,
                "chosen_agent": "NONE",
                "agent_outcome": "",
                "error": None
            }
            
            try:
                final_state = await university_agent_graph.ainvoke(initial_state, config=config)
                response = final_state.get('agent_outcome', 'Sorry, I could not process your request.')
                
                if final_state.get('error'):
                    error_message = f"An error occurred: {final_state['error']}"
                    st.error(error_message)
                    response = "Sorry, an internal error occurred. Please try again later."
                
                # Format and display the response
                formatted_response = format_response(response)
                message_placeholder.markdown(formatted_response)
                add_message_to_history("assistant", formatted_response)
                
            except GraphRecursionError as e:
                error_message = "Sorry, the process took too long or got stuck. Please try rephrasing your question."
                st.error(error_message)
                message_placeholder.markdown(error_message)
                add_message_to_history("assistant", error_message)
                
            except Exception as e:
                error_message = f"Sorry, an unexpected error occurred: {str(e)}"
                st.error(error_message)
                message_placeholder.markdown(error_message)
                add_message_to_history("assistant", error_message)
                traceback.print_exc()
                
    except Exception as e:
        st.error(f"Error in process_user_input: {str(e)}")
        traceback.print_exc()
    finally:
        st.session_state.processing = False
        st.rerun()

# Handle user input
if user_input and not st.session_state.processing:
    asyncio.run(process_user_input(user_input))

st.divider()
st.caption("Powered by Pydantic AI, LangGraph, Crawl4AI, Supabase & Streamlit")