"""Main FastAPI application file."""
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Load environment variables
load_dotenv()

# Import the compiled graph
from src.agent.graph import graph

app = FastAPI()

# Serve static files (for CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="templates/static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    """Request model for the user's query."""
    question: str

@app.post("/run")
async def run_graph(request: QueryRequest):
    """Runs the LangGraph agent with the user's question."""
    try:
        # Run the graph asynchronously with the user's question
        config = {"configurable": {"thread_id": "CORPUS_RESEARCH"}}
        response_data = ""
        async for event in graph.astream(
            {"question": request.question}, config=config
        ):
            if "report" in event:
                report_content = event["report"].content
                if isinstance(report_content, list): # Check if content is a list of parts
                    for part in report_content:
                        if hasattr(part, 'text'):
                            response_data += part.text
                        elif isinstance(part, str):
                            response_data += part
                elif isinstance(report_content, str): # Handle plain string content
                    response_data += report_content

        return {"response": response_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serves the main chat page."""
    return templates.TemplateResponse("chat.html", {"request": request}) 