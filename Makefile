format:
	@echo "Formatting code..."
	@uv run black .

repo:
	@echo "Running Repo Agent..."
	@uv run python main.py

judge:
	@echo "Running Judge Agent..."
	@uv run python -m tests.LaaJ.judge

rag:
	@echo "Running RAG Agent..."
	@uv run python -m agents.rag_agent

server:
	@echo "Starting FastAPI server..."
	@uv run uvicorn server:app --port=8000 --reload --reload-exclude=".repos/*" --reload-exclude=".wikis/*"