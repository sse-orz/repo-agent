format:
	@echo "Formatting code..."
	@uv run black .

run:
	@echo "Running Wiki Agent..."
	@uv run python main.py

rag:
	@echo "Running RAG Agent..."
	@uv run python -m agents.rag_agent