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
    result: Any = None  # Will hold {"queries": [...], "mini_report": "..."}
    docs: list = field(default_factory=list)
    references: List[str] = field(default_factory=list) # Changed default from None

@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """
    question: str = ""
    subtasks: List[Subtask] = field(default_factory=list)
    final_report: str = ""
    # changeme: str = "example" # Removed unused field


PLANNING_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a research assistant. Given the following question, break it down into a list of 5 to 7 clear, actionable subtasks that, when answered, will fully address the original question.\nQuestion: {question}\nSubtasks:\n1.
    """
)

QUERY_GEN_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an expert search assistant. Given the following subtask, generate 1 to 3 concise and effective search queries that would help find information to answer the subtask.\nSubtask: {subtask}\nQueries:\n1.
    """
)

MINI_REPORT_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a research assistant. Given the following subtask and the retrieved documents, write a concise, rationalized mini-report that answers the subtask.
    Cite the sources for your information using the provided source URLs. For each piece of information, try to mention the source document number (e.g., [Source 1], [Source 2]).

    Subtask: {subtask}
    
    Documents:
    {documents}
    
    Mini-report:
    """
)

FINAL_REPORT_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a research assistant. Given the original question and the following mini-reports for each subtask, synthesize a comprehensive, rationalized final report that answers the original question. 
    Use in-text citations like [1], [2], etc., corresponding to the provided list of references. Only cite sources from this list.
    
    Question: {question}
    
    Mini-reports:
    {mini_reports}
    
    References for citation:
    {formatted_references}
    
    Final Report:
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
    for subtask in subtasks:
        if subtask.status == "queries_generated" and subtask.result and "queries" in subtask.result:
            queries = subtask.result.get("queries", [])
            all_retrieved_docs = []
            all_references = []
            for query in queries:
                retrieved_docs_for_query = await retriever.aget_relevant_documents(query) # Use async version
                all_retrieved_docs.extend(retrieved_docs_for_query)
                for doc in retrieved_docs_for_query:
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

async def mini_report_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generates a mini-report for the first subtask with docs but no mini-report."""
    subtasks = state.subtasks or []
    for subtask_idx, subtask in enumerate(subtasks):
        if subtask.status == "docs_retrieved" and (not subtask.result or not subtask.result.get("mini_report")):
            docs_for_prompt = []
            for i, doc in enumerate(subtask.docs or []):
                source_url = getattr(doc, 'metadata', {}).get('source', 'N/A')
                docs_for_prompt.append(f"Document [Source {i+1}]: ({source_url})\\n{getattr(doc, 'page_content', str(doc))}\\n---")
            
            docs_text = "\\n".join(docs_for_prompt)
            if not docs_text:
                docs_text = "No documents were retrieved for this subtask."

            prompt = MINI_REPORT_PROMPT.format(subtask=subtask.description, documents=docs_text)
            response = await llm.ainvoke(prompt)
            mini_report = response.content.strip()
            
            if not subtask.result: # Ensure result is a dict
                subtask.result = {}
            subtask.result["mini_report"] = mini_report
            subtask.status = "mini_report_generated"
            break  # Only process one subtask per call
    return {**state.__dict__, "subtasks": subtasks}

async def final_report_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Synthesizes a final report from all mini-reports using the LLM, incorporating in-text citations, and prepares a structured list of references."""
    mini_reports_for_prompt = []
    all_unique_references = set()
    
    for i, subtask in enumerate(state.subtasks or []):
        if subtask.result and subtask.result.get("mini_report"):
            mini_reports_for_prompt.append(f"Subtask {i+1}: {subtask.description}\n{subtask.result['mini_report']}")
        if subtask.references:
            all_unique_references.update(subtask.references)
            
    mini_reports_text = "\n\n".join(mini_reports_for_prompt)
    
    # Prepare formatted list of unique references for the prompt and for structured output
    sorted_references = sorted(list(all_unique_references))
    formatted_references_for_prompt = "\n".join([f"[{i+1}] {url}" for i, url in enumerate(sorted_references)])
    structured_references_for_response = [{ "id": str(i+1), "url": url } for i, url in enumerate(sorted_references)]
    
    prompt = FINAL_REPORT_PROMPT.format(
        question=state.question, 
        mini_reports=mini_reports_text,
        formatted_references=formatted_references_for_prompt
    )
    
    response = await llm.ainvoke(prompt)
    final_report_content = response.content.strip()
    
    # The agent now returns the LLM-generated report and the structured references
    # The FastAPI app will place these into the final response.
    return {
        **state.__dict__,
        "final_report": final_report_content, 
        "global_references": structured_references_for_response
    }

def router(state: State, config: RunnableConfig) -> str: # Ensure router returns str
    subtasks = state.subtasks or []
    if not subtasks and state.question: # If planning resulted in no subtasks for a valid question
        return "final_report_agent" # Or END if no report is possible/needed

    for subtask in subtasks:
        if subtask.status == "pending":
            return "query_generation_agent"
    for subtask in subtasks:
        if subtask.status == "queries_generated":
            return "retriever_agent"
    for subtask in subtasks:
        if subtask.status == "docs_retrieved" and (not subtask.result or not subtask.result.get("mini_report")):
            return "mini_report_agent"
    
    if all(subtask.status == "mini_report_generated" for subtask in subtasks if subtasks):
        return "final_report_agent"
    
    # Fallback: if there are subtasks but they are not all mini_report_generated and don't match other statuses,
    # it implies something is wrong or the loop is stuck. For safety, could route to END or an error node.
    # However, if the agents correctly update statuses, this path shouldn't be hit often with subtasks.
    # If no subtasks are left (e.g. after planning an empty question), it should go to final_report or END.
    if not state.question: # No initial question
        return END
        
    return "final_report_agent" # Default if logic above doesn't find a specific next step for active subtasks


# Define the graph
graph = StateGraph(State, config_schema=Configuration)

# Add all processing nodes
graph.add_node(planning_agent)
graph.add_node(query_generation_agent)
graph.add_node(retriever_agent)
graph.add_node(mini_report_agent)
graph.add_node(final_report_agent)

# Define the entry point
graph.set_entry_point("planning_agent")

# Conditional Edges using the router
path_map = {
    "query_generation_agent": "query_generation_agent",
    "retriever_agent": "retriever_agent",
    "mini_report_agent": "mini_report_agent",
    "final_report_agent": "final_report_agent",
    END: END
}

graph.add_conditional_edges("planning_agent", router, path_map)
graph.add_conditional_edges("query_generation_agent", router, path_map)
graph.add_conditional_edges("retriever_agent", router, path_map)
graph.add_conditional_edges("mini_report_agent", router, path_map)

# The final report agent is the last step before ending the graph
graph.add_edge("final_report_agent", END)

# Compile the graph
graph = graph.compile(name="Full Research Graph")
