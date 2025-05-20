import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

# Load environment variables
load_dotenv()

# Import the compiled graph
from src.agent.graph import graph

app = FastAPI()

# Mount static directory for JS/CSS if needed
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    question: str

@app.post("/run")
async def run_graph(request: QueryRequest):
    try:
        # Run the graph asynchronously with the user's question
        result = await graph.ainvoke({"question": request.question})
        return {
            "final_report": result.get("final_report", ""),
            "subtasks": [
                {
                    "description": s.description,
                    "mini_report": getattr(s.result, "mini_report", None) if s.result else None,
                    "references": getattr(s, "references", [])
                } for s in result.get("subtasks", [])
            ],
            "global_references": result.get("global_references", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request}) 