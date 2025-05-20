"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, TypedDict, List, Any
from dotenv import load_dotenv
import os

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatAnthropic(api_key=anthropic_api_key, model="claude-3-opus-20240229")
retriever = TavilySearchAPIRetriever(api_key=tavily_api_key)

class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    # my_configurable_param: str # Removed unused parameter


@dataclass
class Subtask:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    description: str
    status: str = "pending"
    result: Any = None  # Will hold {"queries": [...]} after query generation
    docs: list = field(default_factory=list)
    references: List[str] = field(default_factory=list)

@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """
    question: str = ""
    subtasks: List[Subtask] = field(default_factory=list)
    final_report: str = ""
    global_references: List[Dict[str, str]] = field(default_factory=list)
    # changeme: str = "example" # Removed unused field


PLANNING_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a research assistant. Given the following question, break it down into a list of 2 to 4 clear, actionable subtasks that, when answered, will fully address the original question.\nQuestion: {question}\nSubtasks:\n1.
    """
)

QUERY_GEN_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an expert search assistant. Given the following subtask, generate 1 to 3 concise and effective search queries that would help find information to answer the subtask.\nSubtask: {subtask}\nQueries:\n1.
    """
)

async def planning_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Uses the LLM to generate subtasks from the user's question."""
    question = state.question
    if not question.strip():
        return {**state.__dict__, "subtasks": []}
    prompt = PLANNING_PROMPT.format(question=question)
    response = await llm.ainvoke(prompt)
    # Parse the response into a list of subtasks (split on newlines, remove numbering)
    lines = response.content.split("\n")
    subtask_descriptions = []
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() and line[1] in ".)" or line.startswith("-") or line.startswith("*") ):
            # Remove leading number/bullet
            desc = line.lstrip("1234567890. )-* ").strip()
            if desc:
                subtask_descriptions.append(desc)
        elif line:
            subtask_descriptions.append(line)
    subtasks = [Subtask(description=desc, status="pending", result=None, references=[]) for desc in subtask_descriptions if desc]
    return {**state.__dict__, "subtasks": subtasks}

async def query_generation_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generates search queries for the current subtask using the LLM."""
    subtasks = state.subtasks or []
    for subtask in subtasks:
        if subtask.status == "pending":
            if not subtask.result: # Ensure result is a dict
                subtask.result = {}
            prompt = QUERY_GEN_PROMPT.format(subtask=subtask.description)
            response = await llm.ainvoke(prompt)
            # Parse queries from response
            lines = response.content.split("\n")
            queries = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() and line[1] in ".)" or line.startswith("-") or line.startswith("*") ):
                    desc = line.lstrip("1234567890. )-* ").strip()
                    if desc:
                        queries.append(desc)
                elif line: # Handle cases where LLM might not number/bullet point if only one query
                    queries.append(line)
            
            subtask.result["queries"] = queries
            subtask.status = "queries_generated"
            break  # Only process one subtask for now
    return {**state.__dict__, "subtasks": subtasks}

async def retriever_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Uses Tavily to retrieve documents for the queries of the first subtask with queries."""
    subtasks = state.subtasks or []
    for subtask_idx, subtask in enumerate(subtasks):
        if subtask.status == "queries_generated" and subtask.result and "queries" in subtask.result:
            queries = subtask.result.get("queries", [])
            all_retrieved_docs = []
            all_references = []
            for query_idx, query in enumerate(queries):
                retrieved_docs_for_query = await retriever.ainvoke(query)
                all_retrieved_docs.extend(retrieved_docs_for_query)
                for doc_idx, doc in enumerate(retrieved_docs_for_query):
                    if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                        all_references.append(doc.metadata['source'])
            
            # Remove duplicate documents based on page_content, preserving order
            unique_docs = []
            seen_content = set()
            for doc in all_retrieved_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)
            
            subtask.docs = unique_docs
            subtask.references = list(set(all_references)) # Store unique references
            subtask.status = "docs_retrieved"
            break  # Only process one subtask for now
    return {**state.__dict__, "subtasks": subtasks}

async def final_report_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Synthesizes a final report from all subtasks' docs using the LLM, incorporating in-text citations, and prepares a structured list of references."""
    all_unique_references = set()
    all_docs = []
    if state.subtasks:
        for i, subtask in enumerate(state.subtasks):
            if subtask.docs:
                all_docs.extend(subtask.docs)
            if subtask.references:
                all_unique_references.update(subtask.references)
    
    # Prepare docs text for the prompt
    docs_text = "\n".join([
        f"[Source {i+1}] {getattr(doc, 'metadata', {}).get('source', 'N/A')}\n{getattr(doc, 'page_content', str(doc))}" for i, doc in enumerate(all_docs)
    ])
    # Prepare formatted list of unique references for the prompt and for structured output
    sorted_references = sorted(list(all_unique_references))
    formatted_references_for_prompt = "\n".join([f"[{i+1}] {url}" for i, url in enumerate(sorted_references)])
    
    structured_references_for_response = [{ "id": str(i+1), "url": url } for i, url in enumerate(sorted_references)]
    
    prompt = f"""You are a research assistant. Given the original question and the following retrieved documents, synthesize a comprehensive, rationalized final report that answers the original question. Use in-text citations like [1], [2], etc., corresponding to the provided list of references. Only cite sources from this list.\n\nQuestion: {state.question}\n\nDocuments:\n{docs_text}\n\nReferences for citation:\n{formatted_references_for_prompt}\n\nFinal Report:"""
    response = await llm.ainvoke(prompt)
    final_report_content = response.content.strip()
    return {
        **state.__dict__,
        "final_report": final_report_content,
        "global_references": structured_references_for_response
    }

def router(state: State, config: RunnableConfig) -> str:
    subtasks = state.subtasks or []
    if not subtasks and state.question:
        return "final_report_agent"
    for subtask in subtasks:
        if subtask.status == "pending":
            return "query_generation_agent"
    for subtask in subtasks:
        if subtask.status == "queries_generated":
            return "retriever_agent"
    if all(subtask.status == "docs_retrieved" for subtask in subtasks if subtasks):
        return "final_report_agent"
    if not state.question:
        return END
    return "final_report_agent"


# Define the graph
graph = StateGraph(State, config_schema=Configuration)
graph.add_node(planning_agent)
graph.add_node(query_generation_agent)
graph.add_node(retriever_agent)
graph.add_node(final_report_agent)
graph.set_entry_point("planning_agent")

# Conditional Edges using the router
path_map = {
    "query_generation_agent": "query_generation_agent",
    "retriever_agent": "retriever_agent",
    "final_report_agent": "final_report_agent",
    END: END
}

graph.add_conditional_edges("planning_agent", router, path_map)
graph.add_conditional_edges("query_generation_agent", router, path_map)
graph.add_conditional_edges("retriever_agent", router, path_map)

# The final report agent is the last step before ending the graph
graph.add_edge("final_report_agent", END)

# Compile the graph
graph = graph.compile(name="Simplified Research Graph")
