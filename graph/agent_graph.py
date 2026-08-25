# graph/agent_graph.py
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.config import RunnableConfig
from typing import Annotated, Dict, List, Any, Literal
from typing_extensions import TypedDict
import logfire
import asyncio
import sys
import os
import traceback # Import traceback for detailed error printing
from pathlib import Path

# Absolute imports
from agents.ovgu_agent import ovgu_expert_agent, OvguAgentDeps
from agents.magdeburg_agent import magdeburg_general_agent, MagdeburgAgentDeps
from agents.fin_agent import fin_expert_agent, FinAgentDeps
from utils import get_model, get_openai_client, get_supabase_client


logfire.configure(send_to_logfire='if-token-present') # Optional

# --- Define Graph State ---
class UniversityAgentState(TypedDict):
    user_query: str
    chosen_agent: Literal["OVGU", "MAGDEBURG", "FIN", "NONE"]
    agent_outcome: str
    error: str | None

# --- Node Functions ---

async def route_query(state: UniversityAgentState) -> Dict[str, Any]:
    """Decide which agent should handle the query."""
    query = state["user_query"].lower()
    print(f"Router: Routing query: '{query}'")
    
    # Define more comprehensive keyword sets
    fin_keywords = {
        "fin", "fakultät für informatik", "informatics", "computer science", "dke", 
        "data and knowledge engineering", "digital engineering", "software engineering", 
        "visual computing", "department of informatics", "informatik", "g29", "g30",
        "g31", "g32", "g33", "g34", "g35", "g36", "g37", "g38", "g39", "g40",
        "prof.", "professor", "lecturer", "faculty", "research group", "research team",
        "hcai", "human-computer interaction", "artificial intelligence", "master program",
        "bachelor program", "course", "module", "curriculum", "study program"
    }
    
    magdeburg_keywords = {
        "magdeburg", "city", "stadt", "sights", "sehenswürdigkeiten", "transport", 
        "verkehr", "events", "veranstaltungen", "elbe", "hbf", "hauptbahnhof", 
        "station", "bahnhof", "leben in magdeburg", "services", "things to do",
        "attractions", "tourist", "tourismus", "museum", "park", "restaurant",
        "cafe", "shopping", "hotel", "accommodation"
    }
    
    ovgu_keywords = {
        "ovgu", "university", "otto von guericke", "uni ", "campus", 
        "student union", "stw", "campus service", "service center", "library",
        "mensa", "cafeteria", "student life", "student services", "admission",
        "application", "enrollment", "registration", "examination", "exam",
        # Moved course/module keywords primarily to FIN, but keep general here
        "lecture", "seminar", "tutorial", "study", "academics"
    }
    
    # Count keyword matches for each category
    fin_matches = sum(1 for keyword in fin_keywords if keyword in query)
    magdeburg_matches = sum(1 for keyword in magdeburg_keywords if keyword in query)
    ovgu_matches = sum(1 for keyword in ovgu_keywords if keyword in query)
    
    # Get the maximum matches
    max_matches = max(fin_matches, magdeburg_matches, ovgu_matches)
    
    # Add a slight bias towards FIN/OVGU if matches are equal, prioritizing FIN
    if max_matches == 0:
        print("Router: Query does not clearly match known topics.")
        return {"chosen_agent": "NONE"}
    
    # Determine the best matching agent with bias
    if fin_matches == max_matches:
        print("Router: Routing to FIN")
        return {"chosen_agent": "FIN"}
    elif ovgu_matches == max_matches:
         # Route to OVGU only if it strictly has more matches than Magdeburg
         # or if FIN didn't win and OVGU >= Magdeburg
        if ovgu_matches > magdeburg_matches or fin_matches < max_matches:
             print("Router: Routing to OVGU")
             return {"chosen_agent": "OVGU"}
        else: # Magdeburg wins tie with OVGU if FIN didn't win
             print("Router: Routing to MAGDEBURG (OVGU tie)")
             return {"chosen_agent": "MAGDEBURG"} 
    elif magdeburg_matches == max_matches:
        print("Router: Routing to MAGDEBURG")
        return {"chosen_agent": "MAGDEBURG"}
    else:
        # Fallback if logic somehow fails (shouldn't happen with max_matches > 0)
        print("Router: Fallback - No agent determined, choosing NONE.")
        return {"chosen_agent": "NONE"}


async def execute_ovgu_agent(state: UniversityAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Executes the OVGU expert agent."""
    print("Node: Executing OVGU Agent")
    query = state["user_query"]
    try:
        supabase_client = get_supabase_client()
        openai_client = get_openai_client()
        deps = OvguAgentDeps(supabase=supabase_client, openai_client=openai_client)
        result = await ovgu_expert_agent.run(query, deps=deps)
        answer = result.data if hasattr(result, 'data') else "OVGU agent did not return structured data."
        print(f"OVGU Agent Result: {str(answer)[:100]}...")
        return {"agent_outcome": str(answer), "error": None}
    except Exception as e:
        print(f"Error in OVGU Agent Node: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return {"agent_outcome": "An error occurred while processing your request about OVGU.", "error": str(e)}


async def execute_magdeburg_agent(state: UniversityAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Executes the Magdeburg general expert agent."""
    print("Node: Executing MAGDEBURG Agent")
    query = state["user_query"]
    try:
        supabase_client = get_supabase_client()
        openai_client = get_openai_client()
        deps = MagdeburgAgentDeps(supabase=supabase_client, openai_client=openai_client)
        result = await magdeburg_general_agent.run(query, deps=deps)
        answer = result.data if hasattr(result, 'data') else "Magdeburg agent did not return structured data."
        print(f"MAGDEBURG Agent Result: {str(answer)[:100]}...")
        return {"agent_outcome": str(answer), "error": None}
    except Exception as e:
        print(f"Error in MAGDEBURG Agent Node: {e}")
        traceback.print_exc()
        return {"agent_outcome": "An error occurred while processing your request about Magdeburg.", "error": str(e)}


async def execute_fin_agent(state: UniversityAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Executes the FIN expert agent."""
    print("Node: Executing FIN Agent")
    query = state["user_query"]
    try:
        supabase_client = get_supabase_client()
        openai_client = get_openai_client()
        deps = FinAgentDeps(supabase=supabase_client, openai_client=openai_client)
        result = await fin_expert_agent.run(query, deps=deps)
        answer = result.data if hasattr(result, 'data') else "FIN agent did not return structured data."
        print(f"FIN Agent Result: {str(answer)[:100]}...")
        return {"agent_outcome": str(answer), "error": None}
    except Exception as e:
        print(f"Error in FIN Agent Node: {e}")
        traceback.print_exc()
        return {"agent_outcome": "An error occurred while processing your request about FIN.", "error": str(e)}


async def handle_no_agent(state: UniversityAgentState) -> Dict[str, Any]:
    """Handles cases where no specific agent could be chosen based on keywords."""
    print("Node: Handling No Agent")
    # Provide a more informative message and suggest topics
    outcome = (
        "I'm sorry, I couldn't determine the specific topic of your question based on keywords. "
        "I can answer questions about:\n"
        "*   **General OVGU topics** (campus, services, student life, administration)\n"
        "*   **Faculty of Informatics (FIN)** (studies, courses, research, staff)\n"
        "*   **City of Magdeburg** (sights, transport, events, services)\n\n"
        "Could you please rephrase your question or specify the topic (OVGU, FIN, or Magdeburg)?"
    )
    return {"agent_outcome": outcome, "error": None}


async def synthesize_response(state: UniversityAgentState) -> Dict[str, Any]:
    """Refine and format the final response based on the chosen agent's output."""
    print("Node: Synthesizing final response")
    agent_outcome = state.get("agent_outcome", "")
    error = state.get("error")

    # In the future, more complex synthesis could happen here.
    # For now, just pass through the outcome, ensuring it's a string.
    final_response = str(agent_outcome) if agent_outcome else "Sorry, I could not generate a response."

    # Ensure error state is preserved if it occurred
    return {"agent_outcome": final_response, "error": error}

# --- Conditional Edge Logic ---
def decide_next_node(state: UniversityAgentState) -> Literal["execute_ovgu_agent", "execute_magdeburg_agent", "execute_fin_agent", "handle_no_agent"]:
    """Determines the next node based on the router's choice."""
    agent = state["chosen_agent"]
    if agent == "OVGU": return "execute_ovgu_agent"
    if agent == "MAGDEBURG": return "execute_magdeburg_agent"
    if agent == "FIN": return "execute_fin_agent"
    return "handle_no_agent"

# --- Build the Graph ---
def build_university_agent_graph():
    """Builds and returns the university agent graph."""
    graph_builder = StateGraph(UniversityAgentState)

    graph_builder.add_node("router", route_query)
    graph_builder.add_node("execute_ovgu_agent", execute_ovgu_agent)
    graph_builder.add_node("execute_magdeburg_agent", execute_magdeburg_agent)
    graph_builder.add_node("execute_fin_agent", execute_fin_agent)
    graph_builder.add_node("handle_no_agent", handle_no_agent)
    graph_builder.add_node("synthesize_response", synthesize_response)

    graph_builder.add_edge(START, "router")

    graph_builder.add_conditional_edges(
        "router",
        decide_next_node,
        {
            "execute_ovgu_agent": "execute_ovgu_agent",
            "execute_magdeburg_agent": "execute_magdeburg_agent",
            "execute_fin_agent": "execute_fin_agent",
            "handle_no_agent": "handle_no_agent"
        }
    )

    graph_builder.add_edge("execute_ovgu_agent", "synthesize_response")
    graph_builder.add_edge("execute_magdeburg_agent", "synthesize_response")
    graph_builder.add_edge("execute_fin_agent", "synthesize_response")
    graph_builder.add_edge("handle_no_agent", "synthesize_response")

    graph_builder.add_edge("synthesize_response", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph

# Compile graph globally so it can be imported by Streamlit
university_agent_graph = build_university_agent_graph()


# --- Example Invocation (for testing when run directly using python -m graph.agent_graph) ---
async def run_graph_example(graph_instance): # Pass the compiled graph instance
    config = {"configurable": {"thread_id": "test-thread-module-run-1"}}
    queries = [
        "What are the opening hours for the OVGU Mensa?",
        "Tell me about the Magdeburg Cathedral.",
        "What is the application deadline for the DKE master at FIN?",
        "How can I get from the Hauptbahnhof to the Elbauenpark?",
        "Where is the OVGU library?",
        "Recommend a good restaurant in Magdeburg.",
        "What's the capital of Germany?"
    ]
    for query in queries:
        print(f"\n--- Testing Query: '{query}' ---")
        initial_state = {"user_query": query, "chosen_agent": "NONE", "agent_outcome": "", "error": None}
        try:
            final_state = await graph_instance.ainvoke(initial_state, config)
            print("Final State:", final_state)
            print("Final Answer:", final_state.get('agent_outcome'))
        except Exception as e:
             print(f"Error invoking graph for query '{query}': {e}")
             traceback.print_exc() # Show full error during testing
        await asyncio.sleep(1) # Be nice to APIs


# This block only runs when the script is executed directly via python -m graph.agent_graph
if __name__ == "__main__":
    print("Running agent_graph.py directly as module...")
    # The graph is already compiled globally above
    asyncio.run(run_graph_example(university_agent_graph))
    print("Graph example run finished.")