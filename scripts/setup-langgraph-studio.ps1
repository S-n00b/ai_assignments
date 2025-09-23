# LangGraph Studio Setup Script
# This script sets up LangGraph Studio for the AI Assignments project

Write-Host "🚀 Setting up LangGraph Studio for AI Assignments..." -ForegroundColor Green

# Check if Python virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not activated. Activating..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
}

# Install LangGraph Studio dependencies
Write-Host "📦 Installing LangGraph Studio dependencies..." -ForegroundColor Blue
pip install langgraph-cli
pip install langgraph-studio
pip install langgraph
pip install langchain
pip install langsmith

# Create LangGraph Studio project directory
$langgraphDir = ".\langgraph_studio"
if (-not (Test-Path $langgraphDir)) {
    Write-Host "📁 Creating LangGraph Studio project directory..." -ForegroundColor Blue
    New-Item -ItemType Directory -Path $langgraphDir
}

# Create LangGraph Studio configuration
Write-Host "⚙️  Creating LangGraph Studio configuration..." -ForegroundColor Blue
$configContent = @"
# LangGraph Studio Configuration
version: "1.0"

# Studio settings
studio:
  host: "localhost"
  port: 8083
  mode: "graph"  # or "chat"
  
# LangSmith integration (optional)
langsmith:
  api_key: ""  # Set your LangSmith API key here
  project: "ai-architecture"
  
# Working directory
working_directory: "./langgraph_studio"

# Log level
log_level: "INFO"
"@

$configContent | Out-File -FilePath "$langgraphDir\langgraph.yml" -Encoding UTF8

# Create a sample LangGraph workflow
Write-Host "📝 Creating sample LangGraph workflow..." -ForegroundColor Blue
$workflowContent = @"
from langgraph import StateGraph, END
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if "END" in last_message.content:
        return "end"
    return "continue"

def continue_agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    messages.append(AIMessage(content="I'm continuing the conversation..."))
    return {"messages": messages, "next": "continue"}

def end_agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    messages.append(AIMessage(content="Conversation ended."))
    return {"messages": messages, "next": "end"}

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("continue", continue_agent)
workflow.add_node("end", end_agent)

# Add edges
workflow.add_edge("continue", "end")
workflow.add_edge("end", END)

# Set entry point
workflow.set_entry_point("continue")

# Compile the workflow
app = workflow.compile()

if __name__ == "__main__":
    # Example usage
    result = app.invoke({"messages": [HumanMessage(content="Hello!")], "next": "continue"})
    print(result)
"@

$workflowContent | Out-File -FilePath "$langgraphDir\sample_workflow.py" -Encoding UTF8

# Create LangGraph Studio startup script
Write-Host "🔧 Creating LangGraph Studio startup script..." -ForegroundColor Blue
$startupScript = @"
@echo off
echo Starting LangGraph Studio...
cd /d "%~dp0"
langgraph dev --host localhost --port 8083 --log-level INFO
pause
"@

$startupScript | Out-File -FilePath "$langgraphDir\start-studio.bat" -Encoding ASCII

Write-Host "✅ LangGraph Studio setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Yellow
Write-Host "1. Start LangGraph Studio: cd langgraph_studio && langgraph dev --host localhost --port 8083" -ForegroundColor White
Write-Host "2. Access LangGraph Studio at: http://localhost:8083" -ForegroundColor White
Write-Host "3. Access via unified platform at: http://localhost:8080/iframe/langgraph-studio" -ForegroundColor White
Write-Host ""
Write-Host "🔗 Useful commands:" -ForegroundColor Yellow
Write-Host "  Start Studio: langgraph dev --host localhost --port 8083" -ForegroundColor White
Write-Host "  Stop Studio: Ctrl+C" -ForegroundColor White
Write-Host "  Check status: curl http://localhost:8083/health" -ForegroundColor White
