format:
	@echo "Formatting code..."
	@uv run black .

repo:
	@echo "Running Repo Agent..."
	@uv run python main.py

rag:
	@echo "Running RAG Agent..."
	@uv run python -m agents.rag_agent

server:
	@echo "Starting FastAPI server..."
	@uv run uvicorn server:app --port 8000 --reload