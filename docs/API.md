# API Interface Documentation

## Overview

This API provides:

- Repository wiki documentation generation and management
- Streaming generation progress updates
- Listing generated wikis
- Repository-related question answering (RAG)

## Basic Information

- Base URL (Agents): `http://localhost:8000/api/v1/agents`
- Base URL (RAG): `http://localhost:8000/api/v1/rag`
- Authentication: No authentication required
- Data Format: JSON (RAG streaming uses SSE)

## Endpoints

### 1. Generate Agent Documentation

**POST** `/generate`

Generate or update the agent documentation for a repository.

#### Request Body

```json
{
  "owner": "string", // Repository owner (required)
  "repo": "string", // Repository name (required)
  "platform": "github", // Platform (optional, default: "github")
  "need_update": false, // Whether to update existing documentation (optional, default: false)
  "branch_mode": "all", // Branch mode: "all", "code", "repo", "check" (optional, default: "all")
  "mode": "fast", // Generation mode: "fast" or "smart" (optional, default: "fast")
  "max_workers": 50, // Maximum number of workers (optional, default: 50)
  "log": false // Enable logging (optional, default: false)
}
```

#### Response

```json
{
  "message": "string", // Response message
  "code": 200, // Status code
  "data": {
    "owner": "string",
    "repo": "string",
    "wiki_path": "string",
    "wiki_url": "string",
    "files": [
      {
        "name": "string",
        "path": "string",
        "url": "string",
        "size": 0
      }
    ],
    "total_files": 0
  }
}
```

### 2. Stream Generate Agent Documentation

**POST** `/generate-stream`

Generate or update the agent documentation for a repository in streaming mode, supporting real-time progress updates.

#### Request Body

Same as `/generate` endpoint.

#### Response

Server-Sent Events (SSE) stream, containing progress updates and final results.

Event format:

```
data: {"message": "string", "code": 200, "data": {...}}
```

### 3. List Generated Documentation

**GET** `/list`

Get the list of all generated wiki documentation.

#### Response

```json
{
  "message": "string",
  "code": 200,
  "data": {
    "wikis": [
      {
        "owner": "string",
        "repo": "string",
        "wiki_path": "string",
        "wiki_url": "string",
        "total_files": 0,
        "generated_at": "string"
      }
    ],
    "total_wikis": 0
  }
}
```

### 4. RAG: Ask Question About Repository

**POST** `/rag/ask`

Ask a question about a specific repository using Retrieval-Augmented Generation (RAG).

#### Request Body

```json
{
  "owner": "string", // Repository owner (required)
  "repo": "string", // Repository name (required)
  "platform": "github", // Platform (optional, default: "github")
  "question": "string" // User question about the repository (required)
}
```

#### Response

```json
{
  "message": "RAG query executed successfully.",
  "code": 200,
  "data": {
    "answer": "string" // Model-generated answer based on repository wiki and code
  }
}
```

### 5. RAG: Stream Question Answering

**POST** `/rag/ask-stream`

Ask a question about a specific repository with streaming RAG answers using SSE.

#### Request Body

Same as `/rag/ask` endpoint.

#### Response

Server-Sent Events (SSE) stream, containing incremental or updated answers.

Event format:

````
data: {
  "message": "RAG update",
  "code": 200,
  "data": {
    "answer": "string",
    "node": "string" // Current node name, e.g. "Intent" / "Rewrite" / "Retrieve" / "Judge" / "Generate"
  }
}
```*** End Patch

> Note:
> - The `answer` field may be empty for intermediate events, and will contain the latest generated answer when available.
> - The `node` field indicates the current node in the RAG graph, which can be used to显示进度（例如正在意图识别 / 重写问题 / 检索 / 评估 / 生成回答等）。

## Error Handling

All endpoints return JSON responses containing error information when errors occur, with the `code` field being non-200.

## Examples

### Generate Documentation

```bash
curl -X POST http://localhost:8000/api/v1/agents/generate \
  -H "Content-Type: application/json" \
  -d '{"owner": "octocat", "repo": "Hello-World"}'
````

### Stream Generate

```bash
curl -N http://localhost:8000/api/v1/agents/generate-stream \
  -X POST -H "Content-Type: application/json" \
  -d '{"owner": "octocat", "repo": "Hello-World"}'
```

### List Documentation

```bash
curl http://localhost:8000/api/v1/agents/list
```

### RAG Ask

```bash
curl -X POST http://localhost:8000/api/v1/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "octocat",
    "repo": "Hello-World",
    "platform": "github",
    "question": "What does this repository do?"
  }'
```

### RAG Ask (Stream)

```bash
curl -N http://localhost:8000/api/v1/rag/ask-stream \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "octocat",
    "repo": "Hello-World",
    "platform": "github",
    "question": "Where is the main entrypoint of this project?"
  }'
```
