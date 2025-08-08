"""
Unit tests for CourseSearchTool
Tests the search functionality in isolation using mocked dependencies
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from test_fixtures import (
    MockData,
    MockVectorStore,
    assert_search_results_format,
    print_test_section,
)
from vector_store import SearchResults


class TestCourseSearchTool(unittest.TestCase):
    """Unit tests for CourseSearchTool functionality"""

    def setUp(self):
        """Set up test environment"""
        print_test_section("CourseSearchTool Unit Tests")

    def test_tool_definition(self):
        """Test that tool definition is correctly formatted"""
        print("\n🔍 Testing tool definition...")

        mock_store = MockVectorStore()
        tool = CourseSearchTool(mock_store)

        definition = tool.get_tool_definition()

        # Check required fields
        self.assertIn("name", definition)
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)

        # Check specific values
        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("query", definition["input_schema"]["properties"])
        self.assertIn("query", definition["input_schema"]["required"])

        print("✅ Tool definition correctly formatted")
        print(f"   - Name: {definition['name']}")
        print(f"   - Required params: {definition['input_schema']['required']}")

    def test_execute_with_valid_data(self):
        """Test execute method with valid data"""
        print("\n🔍 Testing execute with valid data...")

        mock_store = MockVectorStore(return_data=True)
        tool = CourseSearchTool(mock_store)

        result = tool.execute("test query")

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        assert_search_results_format(result)

        # Check that sources are tracked
        self.assertTrue(hasattr(tool, "last_sources"))
        self.assertGreater(len(tool.last_sources), 0)

        print("✅ Execute with valid data successful")
        print(f"   - Result length: {len(result)}")
        print(f"   - Sources tracked: {len(tool.last_sources)}")
        print(f"   - Result preview: {result[:100]}...")

    def test_execute_with_empty_results(self):
        """Test execute method when no results found"""
        print("\n🔍 Testing execute with empty results...")

        mock_store = MockVectorStore(return_data=False)
        tool = CourseSearchTool(mock_store)

        result = tool.execute("nonexistent query")

        self.assertIn("No relevant content found", result)
        self.assertEqual(len(tool.last_sources), 0)

        print("✅ Execute with empty results handled correctly")
        print(f"   - Result: {result}")

    def test_execute_with_error(self):
        """Test execute method when vector store returns error"""
        print("\n🔍 Testing execute with error...")

        mock_store = MockVectorStore(return_error=True)
        tool = CourseSearchTool(mock_store)

        result = tool.execute("test query")

        self.assertIn("error", result.lower())

        print("✅ Execute with error handled correctly")
        print(f"   - Error result: {result}")

    def test_execute_with_course_filter(self):
        """Test execute method with course name filter"""
        print("\n🔍 Testing execute with course filter...")

        mock_store = MockVectorStore(return_data=True)
        tool = CourseSearchTool(mock_store)

        # Mock the search method to verify filters are passed
        original_search = mock_store.search
        search_calls = []

        def mock_search(*args, **kwargs):
            search_calls.append(kwargs)
            return original_search(*args, **kwargs)

        mock_store.search = mock_search

        result = tool.execute("test query", course_name="Computer Use")

        # Verify search was called with correct parameters
        self.assertEqual(len(search_calls), 1)
        self.assertEqual(search_calls[0]["course_name"], "Computer Use")

        print("✅ Execute with course filter successful")
        print(f"   - Course filter applied: {search_calls[0]['course_name']}")

    def test_execute_with_lesson_filter(self):
        """Test execute method with lesson number filter"""
        print("\n🔍 Testing execute with lesson filter...")

        mock_store = MockVectorStore(return_data=True)
        tool = CourseSearchTool(mock_store)

        # Mock the search method to verify filters are passed
        original_search = mock_store.search
        search_calls = []

        def mock_search(*args, **kwargs):
            search_calls.append(kwargs)
            return original_search(*args, **kwargs)

        mock_store.search = mock_search

        result = tool.execute("test query", lesson_number=2)

        # Verify search was called with correct parameters
        self.assertEqual(len(search_calls), 1)
        self.assertEqual(search_calls[0]["lesson_number"], 2)

        print("✅ Execute with lesson filter successful")
        print(f"   - Lesson filter applied: {search_calls[0]['lesson_number']}")

    def test_execute_with_both_filters(self):
        """Test execute method with both course and lesson filters"""
        print("\n🔍 Testing execute with both filters...")

        mock_store = MockVectorStore(return_data=True)
        tool = CourseSearchTool(mock_store)

        # Mock the search method to verify filters are passed
        original_search = mock_store.search
        search_calls = []

        def mock_search(*args, **kwargs):
            search_calls.append(kwargs)
            return original_search(*args, **kwargs)

        mock_store.search = mock_search

        result = tool.execute("test query", course_name="RAG", lesson_number=1)

        # Verify search was called with correct parameters
        self.assertEqual(len(search_calls), 1)
        self.assertEqual(search_calls[0]["course_name"], "RAG")
        self.assertEqual(search_calls[0]["lesson_number"], 1)

        print("✅ Execute with both filters successful")
        print(f"   - Course filter: {search_calls[0]['course_name']}")
        print(f"   - Lesson filter: {search_calls[0]['lesson_number']}")

    def test_format_results(self):
        """Test the _format_results method"""
        print("\n🔍 Testing results formatting...")

        mock_store = MockVectorStore()
        tool = CourseSearchTool(mock_store)

        # Create test search results
        test_results = MockData.SAMPLE_SEARCH_RESULTS

        formatted = tool._format_results(test_results)

        # Check formatting
        self.assertIsInstance(formatted, str)
        self.assertGreater(len(formatted), 0)

        # Check that course titles and lesson numbers are included
        self.assertIn("Building Towards Computer Use", formatted)
        self.assertIn("Lesson", formatted)

        # Check that sources are tracked
        self.assertGreater(len(tool.last_sources), 0)

        print("✅ Results formatting successful")
        print(f"   - Formatted length: {len(formatted)}")
        print(f"   - Sources generated: {len(tool.last_sources)}")

    def test_source_tracking(self):
        """Test that sources are properly tracked"""
        print("\n🔍 Testing source tracking...")

        mock_store = MockVectorStore(return_data=True)
        tool = CourseSearchTool(mock_store)

        # Execute search
        result = tool.execute("test query")

        # Check sources
        sources = tool.last_sources
        self.assertIsInstance(sources, list)
        self.assertGreater(len(sources), 0)

        # Check source structure
        first_source = sources[0]
        self.assertIn("display", first_source)
        self.assertIn("link", first_source)

        print("✅ Source tracking successful")
        print(f"   - Sources count: {len(sources)}")
        print(f"   - First source: {first_source}")

    def test_source_reset(self):
        """Test that sources can be reset"""
        print("\n🔍 Testing source reset...")

        mock_store = MockVectorStore(return_data=True)
        tool = CourseSearchTool(mock_store)

        # Execute search to populate sources
        tool.execute("test query")
        initial_source_count = len(tool.last_sources)

        # Reset sources
        tool.last_sources = []

        self.assertEqual(len(tool.last_sources), 0)

        print("✅ Source reset successful")
        print(f"   - Initial sources: {initial_source_count}")
        print(f"   - After reset: {len(tool.last_sources)}")

    def test_error_message_formatting(self):
        """Test error message formatting with filters"""
        print("\n🔍 Testing error message formatting...")

        mock_store = MockVectorStore(return_data=False)
        tool = CourseSearchTool(mock_store)

        # Test with course filter
        result1 = tool.execute("test query", course_name="NonExistent")
        self.assertIn("in course 'NonExistent'", result1)

        # Test with lesson filter
        result2 = tool.execute("test query", lesson_number=99)
        self.assertIn("in lesson 99", result2)

        # Test with both filters
        result3 = tool.execute(
            "test query", course_name="NonExistent", lesson_number=99
        )
        self.assertIn("in course 'NonExistent'", result3)
        self.assertIn("in lesson 99", result3)

        print("✅ Error message formatting successful")
        print(f"   - Course filter error: {result1}")
        print(f"   - Lesson filter error: {result2}")
        print(f"   - Both filters error: {result3}")

    def test_lesson_link_integration(self):
        """Test integration with lesson link retrieval"""
        print("\n🔍 Testing lesson link integration...")

        mock_store = MockVectorStore(return_data=True)
        tool = CourseSearchTool(mock_store)

        # Execute search
        result = tool.execute("test query")

        # Check that lesson links are attempted to be retrieved
        # (The mock will return None, but the call should be made)
        sources = tool.last_sources

        for source in sources:
            self.assertIn("link", source)
            # Link should be None from mock, but key should exist

        print("✅ Lesson link integration tested")
        print(f"   - Sources with link keys: {len(sources)}")


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)
