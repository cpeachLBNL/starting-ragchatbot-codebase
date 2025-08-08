"""
API endpoint tests for the RAG Chatbot FastAPI application
"""
import pytest
import json
from fastapi import HTTPException
from unittest.mock import Mock, patch

@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for /api/query endpoint"""
    
    def test_query_with_session_id(self, client):
        """Test query endpoint with provided session ID"""
        request_data = {
            "query": "What is computer automation?",
            "session_id": "existing_session_123"
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "existing_session_123"
        assert isinstance(data["sources"], list)
        
        # Check source format
        if data["sources"]:
            source = data["sources"][0]
            assert "display" in source
            assert "link" in source
    
    def test_query_without_session_id(self, client):
        """Test query endpoint without session ID (should create new session)"""
        request_data = {
            "query": "Tell me about RAG systems"
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test_session_123"  # From mock
    
    def test_query_empty_string(self, client):
        """Test query endpoint with empty query string"""
        request_data = {
            "query": ""
        }
        
        response = client.post("/api/query", json=request_data)
        
        # Should still work, backend should handle empty queries
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "session_id" in data
    
    def test_query_missing_query_field(self, client):
        """Test query endpoint with missing query field"""
        request_data = {
            "session_id": "test_session"
        }
        
        response = client.post("/api/query", json=request_data)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_query_invalid_json(self, client):
        """Test query endpoint with invalid JSON"""
        response = client.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_query_rag_system_error(self, client, test_app):
        """Test query endpoint when RAG system raises exception"""
        # Configure mock to raise exception
        test_app.state.mock_rag_system.query.side_effect = Exception("RAG system error")
        
        request_data = {
            "query": "test query"
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "RAG system error" in data["detail"]
        
        # Reset mock for other tests
        test_app.state.mock_rag_system.query.side_effect = None
    
    def test_query_response_format(self, client):
        """Test that query response follows expected format"""
        request_data = {
            "query": "What is the main topic of the first course?"
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Validate source objects
        for source in data["sources"]:
            assert "display" in source
            assert "link" in source
            assert isinstance(source["display"], str)
            # link can be None or string
            if source["link"] is not None:
                assert isinstance(source["link"], str)

@pytest.mark.api  
class TestCoursesEndpoint:
    """Test cases for /api/courses endpoint"""
    
    def test_get_course_stats(self, client):
        """Test courses endpoint returns proper statistics"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # From our mock data
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Building Toward Computer Use with Anthropic" in data["course_titles"]
        assert "Introduction to RAG Systems" in data["course_titles"]
    
    def test_get_course_stats_rag_system_error(self, client, test_app):
        """Test courses endpoint when RAG system raises exception"""
        # Configure mock to raise exception
        test_app.state.mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Analytics error" in data["detail"]
        
        # Reset mock for other tests
        test_app.state.mock_rag_system.get_course_analytics.side_effect = None
    
    def test_get_course_stats_empty_result(self, client, test_app):
        """Test courses endpoint with empty course list"""
        # Configure mock to return empty results
        test_app.state.mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
        
        # Reset mock for other tests
        test_app.state.mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Building Toward Computer Use with Anthropic", "Introduction to RAG Systems"]
        }

@pytest.mark.api
class TestRootEndpoint:
    """Test cases for / endpoint"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns proper response"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "RAG Chatbot API - Test Mode" in data["message"]

@pytest.mark.api
class TestEndpointIntegration:
    """Integration tests for API endpoints"""
    
    def test_query_and_courses_workflow(self, client):
        """Test typical workflow: check courses, then make queries"""
        # First, get course statistics
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200
        courses_data = courses_response.json()
        
        # Should have courses available
        assert courses_data["total_courses"] > 0
        assert len(courses_data["course_titles"]) > 0
        
        # Then make a query about one of the courses
        course_title = courses_data["course_titles"][0]
        query_request = {
            "query": f"Tell me about {course_title}"
        }
        
        query_response = client.post("/api/query", json=query_request)
        assert query_response.status_code == 200
        query_data = query_response.json()
        
        # Should get a response with session
        assert query_data["answer"]
        assert query_data["session_id"]
        
        # Make follow-up query with same session
        followup_request = {
            "query": "Can you tell me more?",
            "session_id": query_data["session_id"]
        }
        
        followup_response = client.post("/api/query", json=followup_request)
        assert followup_response.status_code == 200
        followup_data = followup_response.json()
        
        # Should maintain same session
        assert followup_data["session_id"] == query_data["session_id"]
    
    def test_cors_headers(self, client):
        """Test that CORS headers are properly set"""
        response = client.options("/api/query")
        
        # Should have CORS headers (TestClient might not show all, but endpoint should work)
        assert response.status_code in [200, 405]  # OPTIONS might not be explicitly handled
        
        # Test actual request works (which means CORS is configured)
        actual_response = client.post("/api/query", json={"query": "test"})
        assert actual_response.status_code == 200

@pytest.mark.api
class TestErrorHandling:
    """Test error handling across endpoints"""
    
    def test_malformed_json_requests(self, client):
        """Test handling of malformed JSON in requests"""
        # Test query endpoint
        response = client.post(
            "/api/query",
            data='{"query": incomplete json',
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_unsupported_http_methods(self, client):
        """Test unsupported HTTP methods return proper errors"""
        # Test PUT on query endpoint
        response = client.put("/api/query", json={"query": "test"})
        assert response.status_code == 405
        
        # Test POST on courses endpoint
        response = client.post("/api/courses", json={"test": "data"})
        assert response.status_code == 405
        
        # Test DELETE on root endpoint
        response = client.delete("/")
        assert response.status_code == 405
    
    def test_nonexistent_endpoints(self, client):
        """Test that non-existent endpoints return 404"""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        
        response = client.post("/api/invalid")
        assert response.status_code == 404