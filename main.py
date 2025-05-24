"""Main FastAPI application file."""
import traceback # Keep for debugging if needed, but commented out for now

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.agent.graph import graph

# Load environment variables
load_dotenv()

app = FastAPI()

# Serve static files (for CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    """Request model for the user's query."""
    question: str

@app.post("/run")
async def run_graph(request: QueryRequest):
    """Run the LangGraph agent with the user's question."""
    try:
        config = {"configurable": {"thread_id": "CORPUS_RESEARCH"}}
        final_report_data = "" # Renamed to avoid confusion
        global_references_data = [] # Initialize with a default

        # print(f"Starting graph with question: {request.question}") # Debug log

        async for event in graph.astream(
            {"question": request.question}, config=config
        ):
            # print(f"Received event: {event}") # Debug log
            
            # Check for report content from the 'report' node or 'reporting_agent' node
            report_node_key = None
            if "report" in event and event["report"] is not None: # 'report' is the direct key for the final output typically
                 report_node_key = "report"
            elif "reporting_agent" in event and event["reporting_agent"] is not None: # Check if output is nested under node name
                 report_node_key = "reporting_agent"

            if report_node_key:
                node_output = event[report_node_key]
                current_report_content = ""
                
                # Extract content attribute (e.g. from AIMessage) or use dict access
                if hasattr(node_output, 'content'):
                    current_report_content = node_output.content
                elif isinstance(node_output, dict) and 'content' in node_output:
                    current_report_content = node_output['content']
                elif isinstance(node_output, str): # If the node directly returns a string
                    current_report_content = node_output

                # Accumulate report content if it's a string
                if isinstance(current_report_content, str):
                    final_report_data += current_report_content
                elif isinstance(current_report_content, list): # Handle list of parts
                    for part in current_report_content:
                        if hasattr(part, 'text'):
                            final_report_data += part.text
                        elif isinstance(part, str):
                            final_report_data += part
                
                # Extract global_references if present in this event's node output
                if isinstance(node_output, dict) and "global_references" in node_output:
                    global_references_data = node_output["global_references"]
            
            # Fallback check if global_references comes as a top-level key in an event
            # This might be less common with astream but good to have a check
            if "global_references" in event and event["global_references"] is not None:
                global_references_data = event["global_references"]


        return {
            "final_report": final_report_data,
            "global_references": global_references_data
        }
    except Exception as e:
        print(f"Error in run_graph: {str(e)}") # Debug log
        print(f"Error type: {type(e)}") # Debug log
        traceback.print_exc() # Print full traceback for server-side debugging
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serve the main chat page."""
    return templates.TemplateResponse("chat.html", {"request": request}) 