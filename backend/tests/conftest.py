"""
Pytest configuration and shared fixtures for RAG Chatbot testing
"""
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
import sys
import os
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .test_fixtures import MockData, MockVectorStore, MockAnthropicClient, MockToolManager, create_mock_config

@pytest.fixture
def mock_config():
    """Fixture providing mock configuration"""
    return create_mock_config()

@pytest.fixture
def mock_vector_store():
    """Fixture providing mock vector store with sample data"""
    return MockVectorStore(return_data=True, return_error=False)

@pytest.fixture
def empty_vector_store():
    """Fixture providing empty vector store"""
    return MockVectorStore(return_data=False, return_error=False)

@pytest.fixture
def error_vector_store():
    """Fixture providing vector store that returns errors"""
    return MockVectorStore(return_data=False, return_error=True)

@pytest.fixture
def mock_anthropic_client():
    """Fixture providing mock Anthropic client without tool use"""
    return MockAnthropicClient(simulate_tool_use=False, simulate_error=False)

@pytest.fixture
def mock_anthropic_client_with_tools():
    """Fixture providing mock Anthropic client with tool use"""
    return MockAnthropicClient(simulate_tool_use=True, simulate_error=False)

@pytest.fixture
def error_anthropic_client():
    """Fixture providing mock Anthropic client that raises errors"""
    return MockAnthropicClient(simulate_tool_use=False, simulate_error=True)

@pytest.fixture
def mock_tool_manager():
    """Fixture providing mock tool manager"""
    return MockToolManager(return_error=False)

@pytest.fixture
def error_tool_manager():
    """Fixture providing mock tool manager that returns errors"""
    return MockToolManager(return_error=True)

@pytest.fixture
def sample_courses():
    """Fixture providing sample course data"""
    return MockData.SAMPLE_COURSES

@pytest.fixture
def sample_chunks():
    """Fixture providing sample course chunks"""
    return MockData.SAMPLE_CHUNKS

@pytest.fixture
def sample_search_results():
    """Fixture providing sample search results"""
    return MockData.SAMPLE_SEARCH_RESULTS

@pytest.fixture
def temp_docs_dir():
    """Fixture providing temporary directory with sample documents"""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample course files
    course1_content = """Building Toward Computer Use with Anthropic
Course Title: Building Toward Computer Use with Anthropic
Instructor: Colt Steele

Lesson 0: Introduction
Welcome to this course on computer automation with AI agents.

Lesson 1: API Basics  
In this lesson, we'll learn how to make basic API requests."""
    
    course2_content = """Introduction to RAG Systems
Course Title: Introduction to RAG Systems
Instructor: Andrew Ng

Lesson 1: What is RAG?
RAG stands for Retrieval-Augmented Generation."""
    
    with open(os.path.join(temp_dir, "course1.txt"), "w") as f:
        f.write(course1_content)
        
    with open(os.path.join(temp_dir, "course2.txt"), "w") as f:
        f.write(course2_content)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_app():
    """Fixture providing FastAPI test application without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create test app with same endpoints but without static file mounting
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")
    
    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Import models (same as in app.py)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceObject(BaseModel):
        display: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceObject]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system for testing
    mock_rag_system = Mock()
    mock_rag_system.session_manager.create_session.return_value = "test_session_123"
    mock_rag_system.query.return_value = (
        "This is a test response", 
        [{"display": "Test Course - Lesson 1", "link": "http://test.com"}]
    )
    mock_rag_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Building Toward Computer Use with Anthropic", "Introduction to RAG Systems"]
    }
    
    # API Endpoints (same logic as app.py)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            source_objects = []
            for source in sources:
                if isinstance(source, dict):
                    source_objects.append(SourceObject(
                        display=source.get('display', ''),
                        link=source.get('link')
                    ))
                else:
                    source_objects.append(SourceObject(display=str(source)))
            
            return QueryResponse(
                answer=answer,
                sources=source_objects,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def read_root():
        return {"message": "RAG Chatbot API - Test Mode"}
    
    # Store mock for test access
    app.state.mock_rag_system = mock_rag_system
    
    return app

@pytest.fixture
def client(test_app):
    """Fixture providing FastAPI test client"""
    return TestClient(test_app)

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Auto-applied fixture to set up test environment"""
    # Set test environment variables
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setenv("CHROMA_PATH", "./test_chroma_db")
    
    # Mock external dependencies that might be problematic in tests
    with patch('chromadb.PersistentClient'):
        with patch('sentence_transformers.SentenceTransformer'):
            yield

@pytest.fixture
def mock_session_manager():
    """Fixture providing mock session manager"""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test_session_123"
    mock_manager.add_message.return_value = None
    mock_manager.get_conversation_context.return_value = "Previous conversation context"
    return mock_manager