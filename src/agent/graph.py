"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.retrievers.tavily_search_api import (
    TavilySearchAPIRetriever,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

# Load environment variables from .env file
load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# Constants
PLANNING_MODEL_NAME = "claude-3-opus-20240229"
QUERY_GENERATION_MODEL_NAME = "claude-3-haiku-20240307"
REPORTING_MODEL_NAME = "claude-3-opus-20240229"
MAX_QUERIES_PER_SUBTASK = 3
MAX_RESULTS_PER_QUERY = 3


@dataclass
class SubTask:
    """Represents a subtask in the research process."""
    description: str
    queries: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    mini_report: str = ""
    references: List[Dict[str, str]] = field(default_factory=list)


class State(TypedDict):
    """Represents the state of the research graph."""
    question: str
    subtasks: List[SubTask]
    report: Any # Using Any for report as its structure might vary

# LLM and Tool Initialization
planning_llm = ChatAnthropic(model=PLANNING_MODEL_NAME, temperature=0)
query_generation_llm = ChatAnthropic(model=QUERY_GENERATION_MODEL_NAME, temperature=0)
reporting_llm = ChatAnthropic(model=REPORTING_MODEL_NAME, temperature=0)

retriever_tool = TavilySearchAPIRetriever(k=MAX_RESULTS_PER_QUERY)

# Prompt Templates
planning_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research planning assistant. Your goal is to break down a complex research "
            "question into a series of manageable subtasks. Each subtask should focus on a specific "
            "aspect of the main question. For each subtask, clearly define what needs to be investigated."
            "Do not generate search queries yet."
            "\n\nExample:"
            "Question: What are the latest advancements in renewable energy technology?"
            "Subtasks:"
            "1. Investigate recent breakthroughs in solar panel efficiency and materials."
            "2. Research new developments in wind turbine design and energy storage solutions for wind farms."
            "3. Explore advancements in geothermal energy extraction and utilization."
            "4. Review progress in wave and tidal energy conversion technologies.",
        ),
        ("human", "User's research question: {question}"),
    ]
)

query_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are a query generation assistant. Based on the provided subtask, generate up to "
            f"{MAX_QUERIES_PER_SUBTASK} distinct search queries that will help gather the necessary "
            f"information. Each query should be concise and targeted. Output only the queries, "
            f"each on a new line. Do not number them or add any other text."
            "\n\nExample Subtask: Investigate recent breakthroughs in solar panel efficiency and materials."
            "Example Queries:"
            "latest solar panel efficiency records 2023-2024"
            "new materials for photovoltaic cells research"
            "perovskite solar cell stability advancements",
        ),
        ("human", "Subtask: {subtask_description}"),
    ]
)

reporting_prompt = ChatPromptTemplate.from_template(
    """Role: You are a research report generation assistant.

Task: Synthesize the information gathered from the subtasks into a comprehensive report that answers the user's original research question. The report should be well-structured, coherent, and directly address the question. Include citations for all claims, referencing the documents used. Each citation should be in the format [Source X], where X is the number of the source in the provided list.

User's Question: {question}

Subtask Reports and Documents:
{subtask_reports_and_docs}

Instructions:
1.  Start by directly addressing the user's question.
2.  Organize the report logically, using information from the subtasks as evidence.
3.  Ensure that all factual claims are supported by citations to the provided documents.
4.  List all unique references at the end of the report under a 'References' section. Each reference should include the title and URL.
5.  If no relevant information is found for a subtask, state that explicitly.
6.  If the provided information is insufficient to answer the question, state that and explain what additional information would be needed.
7.  The final output should be a single, markdown-formatted report.

Report:
"""
)


async def planning_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate subtasks from the user's question using an LLM."""
    question = state.question
    if not question.strip():
        return {"subtasks": []}
    planner = planning_prompt | planning_llm
    response = await planner.ainvoke({"question": question}, config=config)
    subtask_descriptions = response.content.strip().split('\n')
    subtasks = [
        SubTask(description=desc.split('. ', 1)[1])
        for desc in subtask_descriptions
        if '. ' in desc
    ]
    return {"subtasks": subtasks}


async def query_generation_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate search queries for the current subtask using an LLM."""
    subtasks = state.subtasks or []
    for subtask_idx, subtask in enumerate(subtasks):
        if not subtask.queries:  # Only generate if not already present
            query_generator = query_generation_prompt | query_generation_llm
            response = await query_generator.ainvoke(
                {"subtask_description": subtask.description},
                config=config
            )
            queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            subtasks[subtask_idx].queries = queries[:MAX_QUERIES_PER_SUBTASK]
            return {"subtasks": subtasks} # Process one subtask at a time
    return {"subtasks": subtasks} # Should not happen if router is correct


async def retriever_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Retrieve documents for the queries of the first subtask with queries using Tavily."""
    subtasks = state.subtasks or []
    for subtask_idx, subtask in enumerate(subtasks):
        if subtask.queries and not subtask.results: # Only retrieve if queries exist and no results yet
            all_docs_for_subtask = []
            all_references_for_subtask = []
            doc_id_counter = 1 # Unique ID for references across the entire process
            if state.get("global_references"):
                doc_id_counter = len(state["global_references"]) + 1

            for query in subtask.queries:
                retrieved_docs = await retriever_tool.ainvoke(query, config=config)
                for doc in retrieved_docs:
                    # Ensure doc is a dictionary and has 'metadata' and 'content'
                    if isinstance(doc, dict) and "metadata" in doc and "content" in doc:
                        title = doc["metadata"].get("title", "N/A")
                        url = doc["metadata"].get("source", "N/A") # Assuming 'source' contains URL
                        content_preview = doc["content"][:200] + "..." # Keep it brief
                        all_docs_for_subtask.append(f"[Source {doc_id_counter}] Title: {title}\nContent: {content_preview}")
                        all_references_for_subtask.append({"id": doc_id_counter, "title": title, "url": url})
                        doc_id_counter += 1
                    elif hasattr(doc, 'metadata') and hasattr(doc, 'page_content'): # Langchain Document like
                        title = doc.metadata.get("title", "N/A")
                        url = doc.metadata.get("source", "N/A")
                        content_preview = doc.page_content[:200] + "..."
                        all_docs_for_subtask.append(f"[Source {doc_id_counter}] Title: {title}\nContent: {content_preview}")
                        all_references_for_subtask.append({"id": doc_id_counter, "title": title, "url": url})
                        doc_id_counter += 1

            subtasks[subtask_idx].results = all_docs_for_subtask
            subtasks[subtask_idx].references = all_references_for_subtask
            return {"subtasks": subtasks} # Process one subtask at a time
    return {"subtasks": subtasks}


async def reporting_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate a final report by synthesizing information from all subtasks."""
    question = state.question
    subtasks = state.subtasks or []

    subtask_reports_and_docs = ""
    global_references = []

    for i, subtask in enumerate(subtasks):
        subtask_reports_and_docs += f"Subtask {i+1}: {subtask.description}\n"
        if subtask.results:
            subtask_reports_and_docs += "Documents:\n" + "\n".join(subtask.results) + "\n\n"
        else:
            subtask_reports_and_docs += "No documents found for this subtask.\n\n"
        global_references.extend(subtask.references)

    # Deduplicate global_references by URL, keeping the first encountered ID
    seen_urls = set()
    unique_references = []
    for ref in global_references:
        if ref["url"] not in seen_urls:
            unique_references.append(ref)
            seen_urls.add(ref["url"])

    report_generator = reporting_prompt | reporting_llm
    response = await report_generator.ainvoke(
        {
            "question": question,
            "subtask_reports_and_docs": subtask_reports_and_docs,
        },
        config=config,
    )
    return {"report": response, "global_references": unique_references}


def router(state: State, config: RunnableConfig) -> str:
    """Determine the next step in the research process."""
    subtasks = state.subtasks or []
    if not subtasks and state.question:
        return "planning_agent"

    for subtask in subtasks:
        if not subtask.queries:
            return "query_generation_agent"
        if not subtask.results:
            return "retriever_agent"

    return "reporting_agent"


# Define the graph
workflow = StateGraph(State)

workflow.add_node("planning_agent", planning_agent)
workflow.add_node("query_generation_agent", query_generation_agent)
workflow.add_node("retriever_agent", retriever_agent)
workflow.add_node("reporting_agent", reporting_agent)

workflow.set_entry_point("planning_agent")

workflow.add_conditional_edges(
    "planning_agent",
    lambda x: "query_generation_agent" if x.get("subtasks") else END,
)
workflow.add_conditional_edges(
    "query_generation_agent",
    lambda x: "retriever_agent" if any(st.queries for st in x.get("subtasks", [])) else "reporting_agent",
    # path_map={ # Not strictly needed if logic is simple
    #     "retriever_agent": "retriever_agent",
    #     "reporting_agent": "reporting_agent"
    # }
)
workflow.add_conditional_edges(
    "retriever_agent",
    # Check if there are more subtasks needing queries or retrieval, or go to report
    lambda x: (
        "query_generation_agent" if any(not st.queries for st in x.get("subtasks", [])) else
        "retriever_agent" if any(st.queries and not st.results for st in x.get("subtasks", [])) else
        "reporting_agent"
    )
)

workflow.add_edge("reporting_agent", END)

graph = workflow.compile()

# Example usage (for testing, can be removed or commented out)
async def main():
    """Run an example of the graph with a sample question."""
    config = {"configurable": {"thread_id": "RESEARCH_THREAD"}}
    example_question = "What are the latest advancements in AI for drug discovery?"

    async for event in graph.astream(
        {"question": example_question}, config=config
    ):
        # import logging
        # logging.info(f"---- EVENT ----\n{event}\n---- END EVENT ----\n")
        if "report" in event:
            # logging.info(f"\n\n--- FINAL REPORT ---\n{event[\'report\'].content}")
            if event.get("global_references"):
                # references_log = "\n--- REFERENCES ---\n"
                # for ref in event["global_references"]:
                #     references_log += f"[{ref['id']}] {ref['title']} ({ref['url']})\n"
                # logging.info(references_log)
                pass # Placeholder for actual logging if needed in the future

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
