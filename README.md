# Semantic Research Pilot

This project is a simple web application that performs semantic information retrieval using LangGraph and Langchain. It takes a user's research question, breaks it down into subtasks, retrieves relevant information using Tavily Search, and synthesizes a rationalized report supported by references.

## Features

*   Web interface for submitting research questions.
*   Multi-step research process orchestrated by LangGraph:
    *   Planning: Breaking down the main question into subtasks.
    *   Query Generation: Creating search queries for each subtask.
    *   Retrieval: Fetching information using Tavily Search API.
    *   Reporting: Generating a final report with citations and a list of references.
*   Dark-themed, modern UI with a collapsible references section.

## Project Structure

```
semantic_research_pilot/
├── src/
│   └── agent/
│       └── graph.py       # Core LangGraph agent logic
├── templates/
│   └── chat.html        # Frontend HTML and JavaScript
├── main.py              # FastAPI application server
├── requirements.txt     # Python dependencies
├── .env.example         # Example for environment variables (create .env from this)
└── README.md            # This file
```

## Setup and Installation

### Prerequisites

*   Python 3.9+ recommended
*   Access to Anthropic API (for Claude model)
*   Access to Tavily Search API

### 1. Clone the Repository (if applicable)

If this project is in a Git repository, clone it:
```bash
git clone <repository_url>
cd semantic_research_pilot
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

*   On Windows:
    ```bash
    .\venv\Scripts\activate
    ```
*   On macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

### 3. Set Up Environment Variables

Copy the `.env.example` file to a new file named `.env`:

```bash
cp .env.example .env  # macOS/Linux
# For Windows, manually create .env and paste content from .env.example
```

Open the `.env` file and add your API keys:

```env
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY_HERE"
TAVILY_API_KEY="YOUR_TAVILY_API_KEY_HERE"
```

### 4. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Application

Once the setup is complete, you can run the FastAPI application using Uvicorn:

```bash
uvicorn main:app --reload
```

The application will typically be available at `http://127.0.0.1:8000` in your web browser.

## Usage

1.  Open your web browser and navigate to the application URL.
2.  Type your research question into the input field.
3.  Press "Send" or hit Enter.
4.  The assistant will process your question and display a report along with a list of references.

## Potential Expansions

*   Integrate more diverse search tools (e.g., ArXiv, Wikipedia).
*   Add different types of agents for specialized tasks (e.g., data analysis, code generation).
*   Implement user accounts and history.
*   Enhance error handling and user feedback.
*   Allow selection of different LLMs.
