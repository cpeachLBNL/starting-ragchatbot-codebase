"""
Test fixtures and mock data for RAG Chatbot testing
"""
import sys
import os
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any, Optional
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults

class MockData:
    """Container for mock data used across tests"""
    
    # Sample course data based on actual course structure
    SAMPLE_COURSES = [
        Course(
            title="Building Towards Computer Use with Anthropic",
            instructor="Colt Steele",
            course_link="https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
            lessons=[
                Lesson(lesson_number=0, title="Introduction", lesson_link="https://learn.deeplearning.ai/lesson/intro"),
                Lesson(lesson_number=1, title="API Basics", lesson_link="https://learn.deeplearning.ai/lesson/basics"),
                Lesson(lesson_number=2, title="Multi-modal Requests", lesson_link="https://learn.deeplearning.ai/lesson/multimodal")
            ]
        ),
        Course(
            title="Introduction to RAG Systems",
            instructor="Andrew Ng",
            course_link="https://www.deeplearning.ai/short-courses/rag/",
            lessons=[
                Lesson(lesson_number=1, title="What is RAG?", lesson_link="https://learn.deeplearning.ai/lesson/rag-intro"),
                Lesson(lesson_number=2, title="Vector Databases", lesson_link="https://learn.deeplearning.ai/lesson/vector-db")
            ]
        )
    ]
    
    # Sample course chunks for testing
    SAMPLE_CHUNKS = [
        CourseChunk(
            content="Welcome to Building Toward Computer Use with Anthropic. This course teaches you about computer automation using AI agents.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="In this lesson, we'll learn how to make basic API requests to Anthropic's Claude API for text generation.",
            course_title="Building Towards Computer Use with Anthropic", 
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="RAG stands for Retrieval-Augmented Generation. It combines information retrieval with text generation to create more accurate and contextual responses.",
            course_title="Introduction to RAG Systems",
            lesson_number=1,
            chunk_index=0
        )
    ]
    
    # Sample search results
    SAMPLE_SEARCH_RESULTS = SearchResults(
        documents=[chunk.content for chunk in SAMPLE_CHUNKS],
        metadata=[{
            'course_title': chunk.course_title,
            'lesson_number': chunk.lesson_number,
            'chunk_index': chunk.chunk_index
        } for chunk in SAMPLE_CHUNKS],
        distances=[0.1, 0.2, 0.3]
    )
    
    # Empty search results
    EMPTY_SEARCH_RESULTS = SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )
    
    # Error search results
    ERROR_SEARCH_RESULTS = SearchResults(
        documents=[],
        metadata=[], 
        distances=[],
        error="Search error: Vector store connection failed"
    )

class MockVectorStore:
    """Mock VectorStore for testing"""
    
    def __init__(self, return_data=True, return_error=False):
        self.return_data = return_data
        self.return_error = return_error
        self.course_catalog = Mock()
        self.course_content = Mock()
        
    def search(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> SearchResults:
        """Mock search method"""
        if self.return_error:
            return MockData.ERROR_SEARCH_RESULTS
        elif self.return_data:
            return MockData.SAMPLE_SEARCH_RESULTS
        else:
            return MockData.EMPTY_SEARCH_RESULTS
    
    def _resolve_course_name(self, course_name: str) -> Optional[str]:
        """Mock course name resolution"""
        for course in MockData.SAMPLE_COURSES:
            if course_name.lower() in course.title.lower():
                return course.title
        return None
        
    def get_lesson_link(self, course_title: str, lesson_number: int) -> Optional[str]:
        """Mock lesson link retrieval"""
        for course in MockData.SAMPLE_COURSES:
            if course.title == course_title:
                for lesson in course.lessons:
                    if lesson.lesson_number == lesson_number:
                        return lesson.lesson_link
        return None
        
    def get_course_count(self) -> int:
        """Mock course count"""
        return len(MockData.SAMPLE_COURSES) if self.return_data else 0
        
    def get_existing_course_titles(self) -> List[str]:
        """Mock course titles"""
        return [course.title for course in MockData.SAMPLE_COURSES] if self.return_data else []
        
    def get_all_courses_metadata(self) -> List[Dict[str, Any]]:
        """Mock course metadata"""
        if not self.return_data:
            return []
        
        metadata = []
        for course in MockData.SAMPLE_COURSES:
            lessons_data = []
            for lesson in course.lessons:
                lessons_data.append({
                    "lesson_number": lesson.lesson_number,
                    "lesson_title": lesson.title,
                    "lesson_link": lesson.lesson_link
                })
            
            metadata.append({
                "title": course.title,
                "instructor": course.instructor,
                "course_link": course.course_link,
                "lessons": lessons_data,
                "lesson_count": len(course.lessons)
            })
        return metadata

class MockAnthropicClient:
    """Mock Anthropic client for testing AI Generator"""
    
    def __init__(self, simulate_tool_use=False, simulate_error=False):
        self.simulate_tool_use = simulate_tool_use
        self.simulate_error = simulate_error
        self.messages = Mock()
    
    class MockResponse:
        def __init__(self, simulate_tool_use=False, simulate_error=False):
            self.simulate_tool_use = simulate_tool_use
            self.simulate_error = simulate_error
            
            if simulate_error:
                raise Exception("API Error: Authentication failed")
                
            if simulate_tool_use:
                self.stop_reason = "tool_use"
                self.content = [MockAnthropicClient.MockToolUseContent()]
            else:
                self.stop_reason = "end_turn"
                self.content = [MockAnthropicClient.MockTextContent()]
    
    class MockTextContent:
        def __init__(self):
            self.text = "This is a mock response from Claude."
            self.type = "text"
    
    class MockToolUseContent:
        def __init__(self):
            self.type = "tool_use"
            self.id = "tool_12345"
            self.name = "search_course_content" 
            self.input = {"query": "test query"}
            
    def create(self, **kwargs):
        """Mock message creation"""
        return MockAnthropicClient.MockResponse(
            simulate_tool_use=self.simulate_tool_use,
            simulate_error=self.simulate_error
        )

class MockToolManager:
    """Mock ToolManager for testing"""
    
    def __init__(self, return_error=False):
        self.return_error = return_error
        self.tools = {"search_course_content": Mock(), "get_course_outline": Mock()}
        self.last_sources = [{"display": "Test Course - Lesson 1", "link": "http://test.com"}]
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Mock tool definitions"""
        return [
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Mock tool execution"""
        if self.return_error:
            return "Tool execution failed"
        return "Mock search results from tool execution"
    
    def get_last_sources(self) -> List[Dict[str, str]]:
        """Mock source retrieval"""
        return self.last_sources
        
    def reset_sources(self):
        """Mock source reset"""
        self.last_sources = []

def create_mock_config():
    """Create a mock configuration object"""
    mock_config = Mock()
    mock_config.ANTHROPIC_API_KEY = "test-api-key"
    mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    mock_config.CHUNK_SIZE = 800
    mock_config.CHUNK_OVERLAP = 100
    mock_config.MAX_RESULTS = 5
    mock_config.MAX_HISTORY = 2
    mock_config.CHROMA_PATH = "./test_chroma_db"
    return mock_config

def assert_search_results_format(result: str):
    """Assert that search results follow expected format"""
    assert isinstance(result, str), "Search result should be a string"
    assert len(result) > 0, "Search result should not be empty"

def assert_course_outline_format(result: str):
    """Assert that course outline follows expected format"""
    assert isinstance(result, str), "Course outline should be a string"
    assert "**Course Title:**" in result, "Course outline should include course title"
    assert "**Instructor:**" in result, "Course outline should include instructor"
    assert "**Lessons:**" in result or "No lesson information" in result, "Course outline should include lessons section"

def print_test_section(section_name: str):
    """Print a formatted test section header"""
    print(f"\n{'='*60}")
    print(f"  {section_name}")
    print(f"{'='*60}")