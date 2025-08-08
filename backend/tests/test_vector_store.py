"""
Unit tests for VectorStore
Tests the vector database functionality and data retrieval
"""
import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .test_fixtures import MockData, print_test_section
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStore(unittest.TestCase):
    """Unit tests for VectorStore functionality"""
    
    def setUp(self):
        """Set up test environment"""
        print_test_section("VectorStore Unit Tests")
        
        # Create temporary directory for test ChromaDB
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_path = os.path.join(self.temp_dir, "test_chroma")
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('chromadb.PersistentClient')
    def test_vector_store_initialization(self, mock_client):
        """Test VectorStore initialization"""
        print("\n🔍 Testing VectorStore initialization...")
        
        # Mock ChromaDB client and collections
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Initialize VectorStore
        store = VectorStore(self.chroma_path, "all-MiniLM-L6-v2", max_results=5)
        
        # Verify initialization
        self.assertIsNotNone(store.course_catalog)
        self.assertIsNotNone(store.course_content)
        self.assertEqual(store.max_results, 5)
        
        print("✅ VectorStore initialization successful")
        print(f"   - ChromaDB path: {self.chroma_path}")
        print(f"   - Max results: {store.max_results}")
    
    @patch('chromadb.PersistentClient')
    def test_search_results_creation(self, mock_client):
        """Test SearchResults creation from ChromaDB results"""
        print("\n🔍 Testing SearchResults creation...")
        
        # Mock ChromaDB results
        chroma_results = {
            'documents': [['doc1', 'doc2', 'doc3']],
            'metadatas': [[
                {'course_title': 'Course 1', 'lesson_number': 1},
                {'course_title': 'Course 2', 'lesson_number': 2},
                {'course_title': 'Course 1', 'lesson_number': 3}
            ]],
            'distances': [[0.1, 0.2, 0.3]]
        }
        
        # Create SearchResults
        results = SearchResults.from_chroma(chroma_results)
        
        # Verify results
        self.assertEqual(len(results.documents), 3)
        self.assertEqual(len(results.metadata), 3)
        self.assertEqual(len(results.distances), 3)
        self.assertIsNone(results.error)
        self.assertFalse(results.is_empty())
        
        print("✅ SearchResults creation successful")
        print(f"   - Documents: {len(results.documents)}")
        print(f"   - Metadata: {len(results.metadata)}")
        print(f"   - Distances: {len(results.distances)}")
    
    def test_search_results_empty(self):
        """Test empty SearchResults"""
        print("\n🔍 Testing empty SearchResults...")
        
        # Create empty results
        empty_results = SearchResults.empty("No results found")
        
        # Verify empty state
        self.assertTrue(empty_results.is_empty())
        self.assertEqual(empty_results.error, "No results found")
        self.assertEqual(len(empty_results.documents), 0)
        
        print("✅ Empty SearchResults handling successful")
        print(f"   - Error message: {empty_results.error}")
        print(f"   - Is empty: {empty_results.is_empty()}")
    
    @patch('chromadb.PersistentClient')
    def test_course_name_resolution(self, mock_client):
        """Test course name resolution using vector search"""
        print("\n🔍 Testing course name resolution...")
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock course catalog query results
        mock_catalog.query.return_value = {
            'documents': [['Building Towards Computer Use with Anthropic']],
            'metadatas': [[{'title': 'Building Towards Computer Use with Anthropic'}]]
        }
        
        store = VectorStore(self.chroma_path, "all-MiniLM-L6-v2")
        
        # Test course name resolution
        resolved = store._resolve_course_name("Computer Use")
        
        self.assertEqual(resolved, "Building Towards Computer Use with Anthropic")
        
        print("✅ Course name resolution successful")
        print(f"   - Query: 'Computer Use'")
        print(f"   - Resolved: '{resolved}'")
    
    @patch('chromadb.PersistentClient')
    def test_course_name_resolution_not_found(self, mock_client):
        """Test course name resolution when course not found"""
        print("\n🔍 Testing course name resolution - not found...")
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock empty query results
        mock_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        store = VectorStore(self.chroma_path, "all-MiniLM-L6-v2")
        
        # Test course name resolution
        resolved = store._resolve_course_name("NonExistentCourse")
        
        self.assertIsNone(resolved)
        
        print("✅ Course name resolution not found handled correctly")
        print(f"   - Query: 'NonExistentCourse'")
        print(f"   - Resolved: {resolved}")
    
    @patch('chromadb.PersistentClient')
    def test_filter_building(self, mock_client):
        """Test filter building for different search parameters"""
        print("\n🔍 Testing filter building...")
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        store = VectorStore(self.chroma_path, "all-MiniLM-L6-v2")
        
        # Test no filters
        filter_none = store._build_filter(None, None)
        self.assertIsNone(filter_none)
        
        # Test course filter only
        filter_course = store._build_filter("Test Course", None)
        self.assertEqual(filter_course, {"course_title": "Test Course"})
        
        # Test lesson filter only  
        filter_lesson = store._build_filter(None, 2)
        self.assertEqual(filter_lesson, {"lesson_number": 2})
        
        # Test both filters
        filter_both = store._build_filter("Test Course", 2)
        expected = {"$and": [
            {"course_title": "Test Course"},
            {"lesson_number": 2}
        ]}
        self.assertEqual(filter_both, expected)
        
        print("✅ Filter building successful")
        print(f"   - No filters: {filter_none}")
        print(f"   - Course only: {filter_course}")
        print(f"   - Lesson only: {filter_lesson}")
        print(f"   - Both filters: {filter_both}")
    
    @patch('chromadb.PersistentClient')
    def test_search_with_filters(self, mock_client):
        """Test search method with various filter combinations"""
        print("\n🔍 Testing search with filters...")
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock successful course resolution
        mock_catalog.query.return_value = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course'}]]
        }
        
        # Mock content search results
        mock_content.query.return_value = {
            'documents': [['Test content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        store = VectorStore(self.chroma_path, "all-MiniLM-L6-v2")
        
        # Test search with course name
        results = store.search("test query", course_name="Test")
        
        self.assertFalse(results.is_empty())
        self.assertIsNone(results.error)
        
        print("✅ Search with filters successful")
        print(f"   - Results count: {len(results.documents)}")
        print(f"   - Error: {results.error}")
    
    @patch('chromadb.PersistentClient')
    def test_search_course_not_found(self, mock_client):
        """Test search when course name cannot be resolved"""
        print("\n🔍 Testing search with non-existent course...")
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock failed course resolution
        mock_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        store = VectorStore(self.chroma_path, "all-MiniLM-L6-v2")
        
        # Test search with non-existent course
        results = store.search("test query", course_name="NonExistent")
        
        self.assertTrue(results.is_empty())
        self.assertIsNotNone(results.error)
        self.assertIn("No course found matching", results.error)
        
        print("✅ Search with non-existent course handled correctly")
        print(f"   - Error: {results.error}")
    
    @patch('chromadb.PersistentClient')
    def test_search_exception_handling(self, mock_client):
        """Test search method exception handling"""
        print("\n🔍 Testing search exception handling...")
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock content search to raise exception
        mock_content.query.side_effect = Exception("Database connection failed")
        
        store = VectorStore(self.chroma_path, "all-MiniLM-L6-v2")
        
        # Test search with exception
        results = store.search("test query")
        
        self.assertTrue(results.is_empty())
        self.assertIsNotNone(results.error)
        self.assertIn("Search error", results.error)
        
        print("✅ Search exception handling successful")
        print(f"   - Error: {results.error}")
    
    @patch('chromadb.PersistentClient')
    def test_add_course_metadata(self, mock_client):
        """Test adding course metadata to catalog"""
        print("\n🔍 Testing add course metadata...")
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        store = VectorStore(self.chroma_path, "all-MiniLM-L6-v2")
        
        # Test adding course metadata
        course = MockData.SAMPLE_COURSES[0]
        store.add_course_metadata(course)
        
        # Verify catalog.add was called
        mock_catalog.add.assert_called_once()
        
        # Check the call arguments
        call_args = mock_catalog.add.call_args
        self.assertIn('documents', call_args.kwargs)
        self.assertIn('metadatas', call_args.kwargs)
        self.assertIn('ids', call_args.kwargs)
        
        print("✅ Add course metadata successful")
        print(f"   - Course: {course.title}")
        print(f"   - Catalog.add called: {mock_catalog.add.called}")
    
    @patch('chromadb.PersistentClient')
    def test_add_course_content(self, mock_client):
        """Test adding course content chunks"""
        print("\n🔍 Testing add course content...")
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        store = VectorStore(self.chroma_path, "all-MiniLM-L6-v2")
        
        # Test adding course content
        chunks = MockData.SAMPLE_CHUNKS
        store.add_course_content(chunks)
        
        # Verify content.add was called
        mock_content.add.assert_called_once()
        
        # Check the call arguments
        call_args = mock_content.add.call_args
        self.assertIn('documents', call_args.kwargs)
        self.assertIn('metadatas', call_args.kwargs)
        self.assertIn('ids', call_args.kwargs)
        
        print("✅ Add course content successful")
        print(f"   - Chunks: {len(chunks)}")
        print(f"   - Content.add called: {mock_content.add.called}")
    
    @patch('chromadb.PersistentClient')
    def test_get_existing_course_titles(self, mock_client):
        """Test retrieving existing course titles"""
        print("\n🔍 Testing get existing course titles...")
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock catalog.get results
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        
        store = VectorStore(self.chroma_path, "all-MiniLM-L6-v2")
        
        # Test getting course titles
        titles = store.get_existing_course_titles()
        
        self.assertEqual(len(titles), 3)
        self.assertIn('Course 1', titles)
        self.assertIn('Course 2', titles)
        self.assertIn('Course 3', titles)
        
        print("✅ Get existing course titles successful")
        print(f"   - Titles: {titles}")
    
    @patch('chromadb.PersistentClient')
    def test_get_course_count(self, mock_client):
        """Test getting course count"""
        print("\n🔍 Testing get course count...")
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock catalog.get results
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2']
        }
        
        store = VectorStore(self.chroma_path, "all-MiniLM-L6-v2")
        
        # Test getting course count
        count = store.get_course_count()
        
        self.assertEqual(count, 2)
        
        print("✅ Get course count successful")
        print(f"   - Count: {count}")


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)