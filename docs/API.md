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
  "mode": "sub", // generation agent: "sub" or "moe"
  "request": {
    "owner": "string", // Repository owner (required)
    "repo": "string", // Repository name (required)
    "platform": "github", // Platform (optional, default: "github")
    "need_update": false, // Whether to update existing documentation (optional, default: false)
    "branch_mode": "all", // Branch mode: "all", "code", "repo", "check" (optional, default: "all")
    "mode": "fast", // Generation mode: "fast" or "smart" (optional, default: "fast")
    "max_workers": 50, // Maximum number of workers (optional, default: 50)
    "log": false // Enable logging (optional, default: false)
  }
}
```

**Mode Parameter:**

- `"sub"`: Use ParentGraphBuilder
- `"moe"`: Use MoeAgent

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

**Progress Stages by Mode:**

_Sub mode (`"mode": "sub"`):_

- `started` (0%): Starting documentation generation
- `basic_info_node` (15%): Processing basic information
- `check_update_node` (35%): Checking for updates
- `repo_info_graph` (65%): Processing repository information
- `code_analysis_graph` (85%): Analyzing code
- `completed` (100%): Documentation generation completed

_Moe mode (`"mode": "moe"`):_

- `started` (0%): Starting MoeAgent documentation generation
- `repo_info` (10%): Collecting repository information
- `file_selection` (25%): Selecting important files
- `module_clustering` (40%): Clustering files into modules
- `module_docs` (70%): Generating module documentation
- `macro_docs` (85%): Generating macro documentation
- `index_generation` (95%): Generating index and summary
- `completed` (100%): Documentation generation completed successfully

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

### 4. Get Wiki Files (Progressive Loading)

**GET** `/wikis/{owner}/{repo}`

Get currently generated wiki files for a repository. This endpoint is used for progressive loading to query files that have been generated so far, without waiting for the entire generation to complete.

#### Path Parameters

- `owner` (string, required): Repository owner
- `repo` (string, required): Repository name

#### Query Parameters

- `mode` (string, optional): Agent mode - `"sub"` (default) or `"moe"`

#### Response

```json
{
  "message": "Success",
  "code": 200,
  "data": {
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

### 5. RAG: Ask Question About Repository

**POST** `/rag/ask`

Ask a question about a specific repository using Retrieval-Augmented Generation (RAG).

#### Request Body

```json
{
  "owner": "string", // Repository owner (required)
  "repo": "string", // Repository name (required)
  "platform": "github", // Platform (optional, default: "github")
  "mode": "fast", // RAG mode: "fast" (default) or "smart"
  "question": "string" // User question about the repository (required)
}
```

#### Response

```json
{
  "message": "RAG query executed successfully.",
  "code": 200,
  "data": {
    "answer": "string", // Model-generated answer based on repository wiki and code
    "sources": ["string"] // Optional list of referenced document sources used to answer
  }
}
```

### 6. RAG: Stream Question Answering

**POST** `/rag/ask-stream`

Ask a question about a specific repository with streaming RAG answers using SSE.

#### Request Body

Same as `/rag/ask` endpoint.

#### Response

Server-Sent Events (SSE) stream, containing incremental or updated answers.

Event format:

```
data: {
  "message": "RAG update",
  "code": 200,
  "data": {
    "answer": "string",
    "node": "string", // Current node name, e.g. "Intent" / "Rewrite" / "Retrieve" / "Judge" / "Generate"
    "sources": ["string"] // Optional list of referenced document sources accumulated so far
  }
}
```

> Note:
> - The `answer` field may be empty for intermediate events, and will contain the latest generated answer when available.
> - The `node` field indicates the current node in the RAG graph, which can be used to show progress (e.g. intent recognition / rewrite question / retrieve / evaluate / generate answer).

## Error Handling

All endpoints return JSON responses containing error information when errors occur, with the `code` field being non-200.

## Examples

### Generate Documentation

```bash
curl -X POST http://localhost:8000/api/v1/agents/generate \
  -H "Content-Type: application/json" \
  -d '{"owner": "octocat", "repo": "Hello-World"}'
```

### Stream Generate

```bash
curl -N http://localhost:8000/api/v1/agents/generate-stream \
  -X POST -H "Content-Type: application/json" \
  -d '{
    "mode": "sub",
    "request": {
      "owner": "facebook",
      "repo": "zstd",
      "platform": "github",
      "need_update": false,
      "branch_mode": "all",
      "mode": "fast",
      "max_workers": 50,
      "log": false
    }
  }'
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
    "owner": "facebook",
    "repo": "zstd",
    "platform": "github",
    "mode": "fast",
    "question": "What does this repository do?"
  }'
```

### Get Wiki Files (Progressive Loading)

```bash
curl "http://localhost:8000/api/v1/agents/wikis/octocat/Hello-World?mode=sub"
```

### RAG Ask (Stream)

```bash
curl -N http://localhost:8000/api/v1/rag/ask-stream \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "facebook",
    "repo": "zstd",
    "platform": "github",
    "mode": "fast",
    "question": "Where is the main entrypoint of this project?"
  }'
```
