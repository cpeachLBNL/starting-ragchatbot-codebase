# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Dependency Management
```bash
# Install/sync dependencies
uv sync

# Add new dependency
uv add package_name
```

### Environment Setup
- Create `.env` file with `ANTHROPIC_API_KEY=your_key_here`
- Application runs on http://localhost:8000
- API docs available at http://localhost:8000/docs

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot system** for querying educational course materials. The system uses a tool-based approach where Claude decides when to search the knowledge base versus answering from general knowledge.

### Core Components

**RAGSystem** (`backend/rag_system.py`) - Main orchestrator that coordinates all components:
- Processes user queries and maintains conversation sessions
- Manages tool execution for course content search
- Handles both direct AI responses and RAG-enhanced responses

**AI Generator** (`backend/ai_generator.py`) - Claude API integration with tool support:
- System prompt optimized for educational content
- Handles tool execution workflow for search operations
- Temperature set to 0 for consistent responses

**Tool-Based Search** (`backend/search_tools.py`) - Implements search functionality:
- `CourseSearchTool` performs semantic search with course/lesson filtering
- `ToolManager` handles tool registration and execution
- Sources are tracked and returned for citation

**Vector Store** (`backend/vector_store.py`) - ChromaDB integration:
- Uses sentence-transformers for embeddings (all-MiniLM-L6-v2)
- Separate collections for course metadata and content chunks
- Smart course name matching and lesson filtering

**Document Processing** (`backend/document_processor.py`) - Text chunking and metadata extraction:
- Sentence-based chunking with configurable size (800 chars) and overlap (100 chars)
- Extracts course titles, lessons, and instructor information from text files
- Supports PDF, DOCX, and TXT formats

### Configuration

All settings in `backend/config.py` loaded from environment variables:
- `CHUNK_SIZE`: 800 characters (text chunk size)
- `CHUNK_OVERLAP`: 100 characters (chunk overlap)
- `MAX_RESULTS`: 5 (search results limit)
- `MAX_HISTORY`: 2 (conversation memory)
- `CHROMA_PATH`: "./chroma_db" (vector store location)

### Query Flow Architecture

The system has two distinct flows:

1. **Direct Response**: General knowledge questions → Claude → Response
2. **RAG-Enhanced Response**: Course-specific questions → Claude tool call → Vector search → Enhanced response

The AI Generator's system prompt instructs Claude to use search tools only for course-specific content and to synthesize results without meta-commentary.

### Data Models

**Course Structure** (`backend/models.py`):
- `Course`: Contains title, lessons, instructor, and links
- `Lesson`: Has lesson number, title, and optional link
- `CourseChunk`: Text chunks with course/lesson metadata for vector storage

### Session Management

**SessionManager** (`backend/session_manager.py`) maintains conversation context:
- Creates unique session IDs for each conversation
- Stores limited message history (configurable via MAX_HISTORY)
- Provides formatted conversation context to AI Generator

### Frontend Integration

**Static Web Interface** (`frontend/`):
- Single-page application with chat interface
- Displays course statistics and collapsible source citations
- Uses fetch API to communicate with FastAPI backend
- Markdown rendering for assistant responses

## Key Implementation Details

### Tool Execution Pattern
The system uses Anthropic's tool calling feature where Claude can invoke `search_course_content` with optional course name and lesson number filters. The tool execution is handled in a two-step process:
1. Initial Claude response with tool use requests
2. Tool execution followed by final response synthesis

### Vector Search Strategy
- Course names support partial matching (e.g., "MCP" matches "Introduction to MCP")
- Lesson filtering by number when specified
- Results include course and lesson context headers
- Source tracking for UI citation display

### Error Handling
- Graceful degradation when no search results found
- UTF-8 encoding fallback with error handling for document processing
- API error propagation with proper HTTP status codes

## Working with Course Content

Course materials are loaded from the `/docs` folder on application startup. The system:
- Automatically processes TXT, PDF, and DOCX files
- Extracts structured course information (titles, lessons, instructors)
- Chunks content for optimal retrieval
- Avoids duplicate processing of existing courses

To add new course materials, place files in the `/docs` folder and restart the application.