# API Interface Documentation

## Overview

This API provides agent documentation generation and management features, supporting repository documentation generation, streaming generation, and documentation list queries.

## Basic Information

- Base URL: `http://localhost:8000/api/v1/agents`
- Authentication: No authentication required
- Data Format: JSON

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
    // Response data
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
  -d '{"owner": "octocat", "repo": "Hello-World"}'
```

### List Documentation

```bash
curl http://localhost:8000/api/v1/agents/list
```
