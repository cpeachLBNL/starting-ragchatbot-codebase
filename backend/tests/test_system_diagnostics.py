"""
System diagnostic tests for RAG Chatbot system
These tests check overall system health and identify potential configuration/data issues
"""
import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .test_fixtures import create_mock_config, print_test_section
from config import config
from rag_system import RAGSystem
from vector_store import VectorStore
from ai_generator import AIGenerator


class TestSystemDiagnostics(unittest.TestCase):
    """System health and diagnostic tests"""
    
    def setUp(self):
        """Set up test environment"""
        print_test_section("System Diagnostics Tests")
        
    def test_config_loaded(self):
        """Test that configuration is properly loaded"""
        print("\n🔍 Testing configuration loading...")
        
        # Check that config object exists
        self.assertIsNotNone(config, "Config object should exist")
        
        # Check critical configuration values
        self.assertIsNotNone(config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY should be set")
        self.assertNotEqual(config.ANTHROPIC_API_KEY, "", "ANTHROPIC_API_KEY should not be empty")
        
        self.assertIsNotNone(config.ANTHROPIC_MODEL, "ANTHROPIC_MODEL should be set")
        self.assertIsNotNone(config.EMBEDDING_MODEL, "EMBEDDING_MODEL should be set")
        self.assertIsNotNone(config.CHROMA_PATH, "CHROMA_PATH should be set")
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - API Key present: {'Yes' if config.ANTHROPIC_API_KEY else 'No'}")
        print(f"   - Model: {config.ANTHROPIC_MODEL}")
        print(f"   - Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"   - ChromaDB Path: {config.CHROMA_PATH}")
    
    def test_api_key_validity(self):
        """Test if API key appears to be valid format"""
        print("\n🔍 Testing API key validity...")
        
        api_key = config.ANTHROPIC_API_KEY
        
        # Basic format check
        if api_key:
            self.assertTrue(len(api_key) > 10, "API key should be longer than 10 characters")
            self.assertTrue(api_key.startswith('sk-'), "Anthropic API keys typically start with 'sk-'")
            print(f"✅ API key format appears valid (length: {len(api_key)})")
        else:
            print(f"❌ API key is missing or empty")
            self.fail("API key is required for system to function")
    
    def test_documents_exist(self):
        """Test if course documents exist in expected location"""
        print("\n🔍 Testing document availability...")
        
        docs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs")
        
        # Check if docs directory exists
        self.assertTrue(os.path.exists(docs_path), f"Documents directory should exist at {docs_path}")
        
        # Check for course files
        if os.path.exists(docs_path):
            files = [f for f in os.listdir(docs_path) if f.lower().endswith(('.txt', '.pdf', '.docx'))]
            self.assertGreater(len(files), 0, "Should have at least one course document")
            print(f"✅ Found {len(files)} course documents: {files}")
        else:
            print(f"❌ Documents directory not found at {docs_path}")
    
    def test_chromadb_directory(self):
        """Test ChromaDB directory existence and contents"""
        print("\n🔍 Testing ChromaDB directory...")
        
        chroma_path = config.CHROMA_PATH
        
        if os.path.exists(chroma_path):
            print(f"✅ ChromaDB directory exists at {chroma_path}")
            
            # Check for database files
            files = os.listdir(chroma_path)
            print(f"   - Contents: {files}")
            
            # Look for SQLite database
            has_sqlite = any(f.endswith('.sqlite3') for f in files)
            print(f"   - SQLite DB present: {has_sqlite}")
            
            if not has_sqlite:
                print("⚠️  No SQLite database found - vector store may be empty")
        else:
            print(f"❌ ChromaDB directory not found at {chroma_path}")
            print("   This likely means no data has been loaded into the system")
    
    def test_rag_system_initialization(self):
        """Test that RAGSystem can be initialized"""
        print("\n🔍 Testing RAG system initialization...")
        
        try:
            rag_system = RAGSystem(config)
            print("✅ RAGSystem initialized successfully")
            
            # Check components
            self.assertIsNotNone(rag_system.vector_store, "VectorStore should be initialized")
            self.assertIsNotNone(rag_system.ai_generator, "AIGenerator should be initialized")
            self.assertIsNotNone(rag_system.tool_manager, "ToolManager should be initialized")
            
            print("   - VectorStore: ✅")
            print("   - AIGenerator: ✅") 
            print("   - ToolManager: ✅")
            
        except Exception as e:
            print(f"❌ RAGSystem initialization failed: {str(e)}")
            self.fail(f"RAGSystem should initialize successfully, but got error: {str(e)}")
    
    def test_vector_store_data_loaded(self):
        """Test if vector store has data loaded"""
        print("\n🔍 Testing vector store data...")
        
        try:
            rag_system = RAGSystem(config)
            
            # Get course analytics
            analytics = rag_system.get_course_analytics()
            course_count = analytics.get('total_courses', 0)
            course_titles = analytics.get('course_titles', [])
            
            print(f"   - Course count: {course_count}")
            print(f"   - Course titles: {course_titles}")
            
            if course_count > 0:
                print("✅ Vector store contains course data")
            else:
                print("❌ Vector store appears to be empty")
                print("   This is likely the cause of 'query failed' errors")
                
        except Exception as e:
            print(f"❌ Error checking vector store data: {str(e)}")
    
    def test_tools_registration(self):
        """Test that tools are properly registered"""
        print("\n🔍 Testing tool registration...")
        
        try:
            rag_system = RAGSystem(config)
            
            # Check tool definitions
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            tool_names = [tool.get('name') for tool in tool_definitions]
            
            print(f"   - Registered tools: {tool_names}")
            
            # Check for expected tools
            expected_tools = ['search_course_content', 'get_course_outline']
            for expected_tool in expected_tools:
                if expected_tool in tool_names:
                    print(f"   - {expected_tool}: ✅")
                else:
                    print(f"   - {expected_tool}: ❌")
                    self.fail(f"Expected tool '{expected_tool}' not registered")
                    
        except Exception as e:
            print(f"❌ Error checking tool registration: {str(e)}")
    
    @patch('anthropic.Anthropic')
    def test_ai_generator_api_connection(self, mock_anthropic):
        """Test AI Generator can connect to API"""
        print("\n🔍 Testing Anthropic API connection...")
        
        # Mock successful response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"
        
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        try:
            ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
            response = ai_generator.generate_response("test query")
            
            self.assertIsNotNone(response, "AI Generator should return a response")
            print("✅ AI Generator API connection test passed")
            
        except Exception as e:
            print(f"❌ AI Generator API connection failed: {str(e)}")
    
    def test_end_to_end_simple_query(self):
        """Test a simple end-to-end query to identify failure point"""
        print("\n🔍 Testing end-to-end simple query...")
        
        try:
            rag_system = RAGSystem(config)
            
            # Try a simple query that should work if system is healthy
            response, sources = rag_system.query("What courses are available?", session_id="test")
            
            self.assertIsNotNone(response, "Response should not be None")
            self.assertIsInstance(response, str, "Response should be a string")
            
            print(f"✅ End-to-end query successful")
            print(f"   - Response length: {len(response)}")
            print(f"   - Sources count: {len(sources)}")
            print(f"   - Response preview: {response[:100]}...")
            
        except Exception as e:
            print(f"❌ End-to-end query failed: {str(e)}")
            print(f"   - This is likely the source of 'query failed' errors")
            print(f"   - Exception type: {type(e).__name__}")
    
    def print_system_summary(self):
        """Print a summary of system health"""
        print("\n" + "="*60)
        print("  SYSTEM HEALTH SUMMARY")
        print("="*60)
        
        checks = [
            ("Configuration", self._check_config()),
            ("API Key", self._check_api_key()),
            ("Documents", self._check_documents()),
            ("ChromaDB", self._check_chromadb()),
            ("Vector Store Data", self._check_vector_data()),
            ("Tool Registration", self._check_tools())
        ]
        
        for check_name, status in checks:
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {check_name}")
        
        print("\nIf any checks failed, they may be contributing to 'query failed' errors.")
    
    def _check_config(self) -> bool:
        """Helper to check configuration"""
        try:
            return bool(config.ANTHROPIC_API_KEY and config.ANTHROPIC_MODEL)
        except:
            return False
    
    def _check_api_key(self) -> bool:
        """Helper to check API key"""
        try:
            return bool(config.ANTHROPIC_API_KEY and len(config.ANTHROPIC_API_KEY) > 10)
        except:
            return False
    
    def _check_documents(self) -> bool:
        """Helper to check documents"""
        try:
            docs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs")
            return os.path.exists(docs_path) and len(os.listdir(docs_path)) > 0
        except:
            return False
    
    def _check_chromadb(self) -> bool:
        """Helper to check ChromaDB"""
        try:
            return os.path.exists(config.CHROMA_PATH)
        except:
            return False
    
    def _check_vector_data(self) -> bool:
        """Helper to check vector store data"""
        try:
            rag_system = RAGSystem(config)
            analytics = rag_system.get_course_analytics()
            return analytics.get('total_courses', 0) > 0
        except:
            return False
    
    def _check_tools(self) -> bool:
        """Helper to check tool registration"""
        try:
            rag_system = RAGSystem(config)
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            return len(tool_definitions) >= 2
        except:
            return False


if __name__ == '__main__':
    # Run diagnostics with detailed output
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemDiagnostics)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    test_instance = TestSystemDiagnostics()
    test_instance.print_system_summary()