"""
Integration tests for RAGSystem
Tests the complete system functionality and component interactions
"""
import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .test_fixtures import (
    create_mock_config, MockAnthropicClient, print_test_section
)
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class TestRAGSystemIntegration(unittest.TestCase):
    """Integration tests for complete RAG system functionality"""
    
    def setUp(self):
        """Set up test environment"""
        print_test_section("RAGSystem Integration Tests")
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock config
        self.mock_config = create_mock_config()
        self.mock_config.CHROMA_PATH = os.path.join(self.temp_dir, "test_chroma")
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('chromadb.PersistentClient')
    @patch('anthropic.Anthropic')
    def test_rag_system_initialization(self, mock_anthropic, mock_chroma):
        """Test complete RAG system initialization"""
        print("\n🔍 Testing RAG system initialization...")
        
        # Setup mocks
        self._setup_chroma_mock(mock_chroma)
        self._setup_anthropic_mock(mock_anthropic)
        
        # Test initialization
        rag_system = RAGSystem(self.mock_config)
        
        # Verify all components are initialized
        self.assertIsNotNone(rag_system.document_processor)
        self.assertIsNotNone(rag_system.vector_store)
        self.assertIsNotNone(rag_system.ai_generator)
        self.assertIsNotNone(rag_system.session_manager)
        self.assertIsNotNone(rag_system.tool_manager)
        
        # Verify tools are registered
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool['name'] for tool in tool_definitions]
        self.assertIn('search_course_content', tool_names)
        self.assertIn('get_course_outline', tool_names)
        
        print("✅ RAG system initialization successful")
        print(f"   - Components initialized: 5")
        print(f"   - Tools registered: {len(tool_definitions)}")
        print(f"   - Tool names: {tool_names}")
    
    @patch('chromadb.PersistentClient')
    @patch('anthropic.Anthropic')
    def test_query_without_tools(self, mock_anthropic, mock_chroma):
        """Test query processing without tool use"""
        print("\n🔍 Testing query without tools...")
        
        # Setup mocks
        self._setup_chroma_mock(mock_chroma)
        
        # Mock Anthropic to return direct response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is a general knowledge answer."
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Test
        rag_system = RAGSystem(self.mock_config)
        response, sources = rag_system.query("What is artificial intelligence?")
        
        # Verify
        self.assertEqual(response, "This is a general knowledge answer.")
        self.assertEqual(len(sources), 0)  # No tools used, so no sources
        
        print("✅ Query without tools successful")
        print(f"   - Response: {response}")
        print(f"   - Sources: {len(sources)}")
    
    @patch('chromadb.PersistentClient') 
    @patch('anthropic.Anthropic')
    def test_query_with_tools(self, mock_anthropic, mock_chroma):
        """Test query processing with tool use"""
        print("\n🔍 Testing query with tools...")
        
        # Setup mocks
        mock_chroma_client = self._setup_chroma_mock(mock_chroma)
        
        # Mock course resolution and search results
        mock_chroma_client.course_catalog.query.return_value = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course'}]]
        }
        
        mock_chroma_client.course_content.query.return_value = {
            'documents': [['This is test course content about AI.']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        # Mock Anthropic to use tools
        mock_client = Mock()
        
        # First response with tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.id = "tool_123"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "AI concepts"}
        mock_tool_response.content = [mock_tool_content]
        
        # Final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Based on the course content, AI is about machine intelligence."
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Test
        rag_system = RAGSystem(self.mock_config)
        response, sources = rag_system.query("What is AI in the course?")
        
        # Verify
        self.assertEqual(response, "Based on the course content, AI is about machine intelligence.")
        self.assertGreater(len(sources), 0)  # Should have sources from tool use
        
        print("✅ Query with tools successful")
        print(f"   - Response: {response}")
        print(f"   - Sources: {len(sources)}")
        print(f"   - API calls made: {mock_client.messages.create.call_count}")
    
    @patch('chromadb.PersistentClient')
    @patch('anthropic.Anthropic')
    def test_session_management(self, mock_anthropic, mock_chroma):
        """Test session management and conversation history"""
        print("\n🔍 Testing session management...")
        
        # Setup mocks
        self._setup_chroma_mock(mock_chroma)
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with context."
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Test
        rag_system = RAGSystem(self.mock_config)
        
        # First query
        response1, sources1 = rag_system.query("What is ML?", session_id="test_session")
        
        # Second query in same session
        response2, sources2 = rag_system.query("Tell me more", session_id="test_session")
        
        # Verify both queries succeeded
        self.assertEqual(response1, "Response with context.")
        self.assertEqual(response2, "Response with context.")
        
        # Verify session history is being used (second call should include history in system prompt)
        self.assertEqual(mock_client.messages.create.call_count, 2)
        
        # Check that second call includes conversation history
        second_call_args = mock_client.messages.create.call_args
        system_prompt = second_call_args.kwargs.get('system', '')
        self.assertIn('Previous conversation', system_prompt)
        
        print("✅ Session management successful")
        print(f"   - Session queries: 2")
        print(f"   - History included: {'Previous conversation' in system_prompt}")
    
    @patch('chromadb.PersistentClient')
    @patch('anthropic.Anthropic')
    def test_course_analytics(self, mock_anthropic, mock_chroma):
        """Test course analytics functionality"""
        print("\n🔍 Testing course analytics...")
        
        # Setup mocks
        mock_chroma_client = self._setup_chroma_mock(mock_chroma)
        
        # Mock course data
        mock_chroma_client.course_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        
        # Test
        rag_system = RAGSystem(self.mock_config)
        analytics = rag_system.get_course_analytics()
        
        # Verify
        self.assertEqual(analytics['total_courses'], 3)
        self.assertEqual(len(analytics['course_titles']), 3)
        self.assertIn('Course 1', analytics['course_titles'])
        
        print("✅ Course analytics successful")
        print(f"   - Total courses: {analytics['total_courses']}")
        print(f"   - Course titles: {analytics['course_titles']}")
    
    @patch('chromadb.PersistentClient')
    @patch('anthropic.Anthropic')
    def test_error_propagation(self, mock_anthropic, mock_chroma):
        """Test error propagation through the system"""
        print("\n🔍 Testing error propagation...")
        
        # Setup mocks
        self._setup_chroma_mock(mock_chroma)
        
        # Mock Anthropic to raise an error
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Authentication failed")
        mock_anthropic.return_value = mock_client
        
        # Test
        rag_system = RAGSystem(self.mock_config)
        
        # Query should raise exception
        with self.assertRaises(Exception) as context:
            response, sources = rag_system.query("test query")
        
        # Verify error message
        self.assertIn("API Authentication failed", str(context.exception))
        
        print("✅ Error propagation successful")
        print(f"   - Error properly propagated: {str(context.exception)}")
    
    @patch('chromadb.PersistentClient')
    @patch('anthropic.Anthropic')
    def test_empty_vector_store(self, mock_anthropic, mock_chroma):
        """Test system behavior with empty vector store"""
        print("\n🔍 Testing empty vector store...")
        
        # Setup mocks with empty data
        mock_chroma_client = self._setup_chroma_mock(mock_chroma)
        
        # Mock empty course catalog
        mock_chroma_client.course_catalog.get.return_value = {'ids': []}
        
        # Mock empty search results
        mock_chroma_client.course_content.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        # Mock tool use response
        mock_client = Mock()
        
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.id = "tool_123"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_content]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "No content found in courses."
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Test
        rag_system = RAGSystem(self.mock_config)
        
        # Check analytics show empty store
        analytics = rag_system.get_course_analytics()
        self.assertEqual(analytics['total_courses'], 0)
        
        # Query should still work but return appropriate response
        response, sources = rag_system.query("What courses are available?")
        self.assertEqual(response, "No content found in courses.")
        
        print("✅ Empty vector store handling successful")
        print(f"   - Course count: {analytics['total_courses']}")
        print(f"   - Response: {response}")
    
    @patch('chromadb.PersistentClient')
    @patch('anthropic.Anthropic')
    def test_tool_execution_failure(self, mock_anthropic, mock_chroma):
        """Test system behavior when tool execution fails"""
        print("\n🔍 Testing tool execution failure...")
        
        # Setup mocks
        mock_chroma_client = self._setup_chroma_mock(mock_chroma)
        
        # Mock ChromaDB to raise exception
        mock_chroma_client.course_content.query.side_effect = Exception("Database connection failed")
        
        # Mock tool use workflow
        mock_client = Mock()
        
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.id = "tool_123"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_content]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Search encountered an error."
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Test
        rag_system = RAGSystem(self.mock_config)
        response, sources = rag_system.query("Search for something")
        
        # Should handle error gracefully
        self.assertEqual(response, "Search encountered an error.")
        
        print("✅ Tool execution failure handling successful")
        print(f"   - Response: {response}")
    
    def _setup_chroma_mock(self, mock_chroma):
        """Helper to setup ChromaDB mocks"""
        mock_client = Mock()
        mock_chroma.return_value = mock_client
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Add references for easier access
        mock_client.course_catalog = mock_catalog
        mock_client.course_content = mock_content
        
        return mock_client
    
    def _setup_anthropic_mock(self, mock_anthropic):
        """Helper to setup Anthropic mocks"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Default response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Mock response"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        return mock_client


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)