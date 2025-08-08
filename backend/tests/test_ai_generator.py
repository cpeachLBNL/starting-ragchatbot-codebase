"""
Unit tests for AIGenerator
Tests the AI generation functionality, tool calling, and API interactions
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from test_fixtures import MockAnthropicClient, MockToolManager, print_test_section


class TestAIGenerator(unittest.TestCase):
    """Unit tests for AIGenerator functionality"""

    def setUp(self):
        """Set up test environment"""
        print_test_section("AIGenerator Unit Tests")
        self.test_api_key = "sk-test-key-12345"
        self.test_model = "claude-sonnet-4-20250514"

    def test_ai_generator_initialization(self):
        """Test AIGenerator initialization"""
        print("\n🔍 Testing AIGenerator initialization...")

        with patch("anthropic.Anthropic") as mock_anthropic:
            ai_gen = AIGenerator(self.test_api_key, self.test_model)

            # Verify initialization
            self.assertEqual(ai_gen.model, self.test_model)
            self.assertIsNotNone(ai_gen.client)
            self.assertIsNotNone(ai_gen.base_params)

            # Check base parameters
            self.assertEqual(ai_gen.base_params["model"], self.test_model)
            self.assertEqual(ai_gen.base_params["temperature"], 0)
            self.assertEqual(ai_gen.base_params["max_tokens"], 800)

        print("✅ AIGenerator initialization successful")
        print(f"   - Model: {ai_gen.model}")
        print(f"   - Temperature: {ai_gen.base_params['temperature']}")
        print(f"   - Max tokens: {ai_gen.base_params['max_tokens']}")

    def test_system_prompt_content(self):
        """Test that system prompt template contains expected content"""
        print("\n🔍 Testing system prompt content...")

        prompt_template = AIGenerator.SYSTEM_PROMPT_TEMPLATE

        # Check for key components
        self.assertIn("search_course_content", prompt_template)
        self.assertIn("get_course_outline", prompt_template)
        self.assertIn("Multi-Round Tool Usage", prompt_template)
        self.assertIn("Course outline queries", prompt_template)
        self.assertIn("Content queries", prompt_template)
        self.assertIn("{max_tool_rounds}", prompt_template)
        self.assertIn("Sequential reasoning", prompt_template)

        print("✅ System prompt content validated")
        print(f"   - Prompt template length: {len(prompt_template)}")
        print(
            f"   - Contains search tool: {'search_course_content' in prompt_template}"
        )
        print(f"   - Contains outline tool: {'get_course_outline' in prompt_template}")
        print(
            f"   - Contains sequential features: {'Sequential reasoning' in prompt_template}"
        )

    @patch("anthropic.Anthropic")
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test generate_response without tools (direct response)"""
        print("\n🔍 Testing generate_response without tools...")

        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is a direct response without tools."
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Test
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        response = ai_gen.generate_response("What is AI?")

        # Verify
        self.assertEqual(response, "This is a direct response without tools.")
        mock_client.messages.create.assert_called_once()

        print("✅ Generate response without tools successful")
        print(f"   - Response: {response}")

    @patch("anthropic.Anthropic")
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test generate_response with conversation history"""
        print("\n🔍 Testing generate_response with conversation history...")

        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with history context."
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Test
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        response = ai_gen.generate_response(
            "Follow up question",
            conversation_history="Previous: What is AI?\nAssistant: AI is artificial intelligence.",
        )

        # Verify call was made
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args

        # Check that history is included in system prompt
        self.assertIn("Previous conversation", call_args.kwargs["system"])

        print("✅ Generate response with conversation history successful")
        print(f"   - Response: {response}")
        print(
            f"   - History included in system: {'Previous conversation' in call_args.kwargs['system']}"
        )

    @patch("anthropic.Anthropic")
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic):
        """Test generate_response with tools available but not used"""
        print("\n🔍 Testing generate_response with tools (no tool use)...")

        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Direct response, no tools needed."
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Test
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        mock_tool_manager = MockToolManager()

        response = ai_gen.generate_response(
            "What is machine learning?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Verify
        self.assertEqual(response, "Direct response, no tools needed.")

        # Check that tools were provided in API call
        call_args = mock_client.messages.create.call_args
        self.assertIn("tools", call_args.kwargs)
        self.assertIn("tool_choice", call_args.kwargs)

        print("✅ Generate response with tools (no tool use) successful")
        print(f"   - Response: {response}")
        print(f"   - Tools provided: {'tools' in call_args.kwargs}")

    @patch("anthropic.Anthropic")
    def test_generate_response_with_tool_use(self, mock_anthropic):
        """Test generate_response with tool use"""
        print("\n🔍 Testing generate_response with tool use...")

        # Setup mocks for tool use workflow
        mock_client = Mock()

        # First response with tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.id = "tool_123"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test query"}
        mock_tool_response.content = [mock_tool_content]

        # Final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Response based on tool results."

        # Setup call sequence
        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]
        mock_anthropic.return_value = mock_client

        # Test
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        mock_tool_manager = MockToolManager()

        response = ai_gen.generate_response(
            "What is covered in the MCP course?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Verify
        self.assertEqual(response, "Response based on tool results.")
        self.assertEqual(mock_client.messages.create.call_count, 2)

        print("✅ Generate response with tool use successful")
        print(f"   - Final response: {response}")
        print(f"   - API calls made: {mock_client.messages.create.call_count}")

    @patch("anthropic.Anthropic")
    def test_helper_methods_functionality(self, mock_anthropic):
        """Test the new helper methods functionality"""
        print("\n🔍 Testing helper methods...")

        # Setup mocks
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator(self.test_api_key, self.test_model)

        # Test _build_system_content
        system_content = ai_gen._build_system_content("Previous conversation", 2)
        self.assertIn("Previous conversation", system_content)
        self.assertIn("up to 2 separate tool calls", system_content)

        # Test _execute_tools
        mock_response = Mock()
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.id = "tool_456"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_response.content = [mock_tool_content]

        mock_tool_manager = MockToolManager()
        tool_results = ai_gen._execute_tools(mock_response, mock_tool_manager)

        self.assertEqual(len(tool_results), 1)
        self.assertEqual(tool_results[0]["type"], "tool_result")
        self.assertEqual(tool_results[0]["tool_use_id"], "tool_456")

        print("✅ Helper methods functionality validated")
        print(
            f"   - System content includes conversation: {'Previous conversation' in system_content}"
        )
        print(f"   - Tool results generated: {len(tool_results)}")

    @patch("anthropic.Anthropic")
    def test_tool_execution_error_handling(self, mock_anthropic):
        """Test error handling during tool execution"""
        print("\n🔍 Testing tool execution error handling...")

        # Setup mocks
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Mock API error
        mock_client.messages.create.side_effect = Exception("API Error")

        # Test
        ai_gen = AIGenerator(self.test_api_key, self.test_model)

        try:
            response = ai_gen.generate_response("test query")
            self.fail("Should have raised an exception")
        except Exception as e:
            self.assertIn("API Error", str(e))

        print("✅ Tool execution error handling successful")
        print(f"   - Exception properly raised")

    @patch("anthropic.Anthropic")
    def test_multiple_tool_calls(self, mock_anthropic):
        """Test handling multiple tool calls in one response"""
        print("\n🔍 Testing multiple tool calls...")

        # Setup mocks
        mock_client = Mock()

        # Response with multiple tool uses
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"

        # Create multiple tool use content blocks
        tool_content_1 = Mock()
        tool_content_1.type = "tool_use"
        tool_content_1.id = "tool_1"
        tool_content_1.name = "search_course_content"
        tool_content_1.input = {"query": "query 1"}

        tool_content_2 = Mock()
        tool_content_2.type = "tool_use"
        tool_content_2.id = "tool_2"
        tool_content_2.name = "get_course_outline"
        tool_content_2.input = {"course_title": "Test Course"}

        mock_tool_response.content = [tool_content_1, tool_content_2]

        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Multiple tools executed."

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]
        mock_anthropic.return_value = mock_client

        # Test
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        mock_tool_manager = MockToolManager()

        response = ai_gen.generate_response(
            "Complex query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Verify
        self.assertEqual(response, "Multiple tools executed.")

        print("✅ Multiple tool calls handling successful")
        print(f"   - Response: {response}")

    @patch("anthropic.Anthropic")
    def test_tool_manager_error(self, mock_anthropic):
        """Test handling when tool manager returns error"""
        print("\n🔍 Testing tool manager error handling...")

        # Setup mocks
        mock_client = Mock()

        # Tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.id = "tool_123"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_content]

        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Tool error handled."

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]
        mock_anthropic.return_value = mock_client

        # Test with error-returning tool manager
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        mock_tool_manager = MockToolManager(return_error=True)

        response = ai_gen.generate_response(
            "test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Verify
        self.assertEqual(response, "Tool error handled.")

        print("✅ Tool manager error handling successful")
        print(f"   - Response: {response}")

    @patch("anthropic.Anthropic")
    def test_api_parameters_construction(self, mock_anthropic):
        """Test API parameters are constructed correctly"""
        print("\n🔍 Testing API parameters construction...")

        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response."
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Test
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        mock_tool_manager = MockToolManager()

        response = ai_gen.generate_response(
            "test query",
            conversation_history="previous context",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Verify API call parameters
        call_args = mock_client.messages.create.call_args
        params = call_args.kwargs

        # Check required parameters
        self.assertIn("model", params)
        self.assertIn("messages", params)
        self.assertIn("system", params)
        self.assertIn("temperature", params)
        self.assertIn("max_tokens", params)
        self.assertIn("tools", params)
        self.assertIn("tool_choice", params)

        # Check values
        self.assertEqual(params["model"], self.test_model)
        self.assertEqual(params["temperature"], 0)
        self.assertEqual(params["max_tokens"], 800)
        self.assertEqual(len(params["messages"]), 1)
        self.assertIn("previous context", params["system"])

        print("✅ API parameters construction successful")
        print(f"   - Model: {params['model']}")
        print(f"   - Temperature: {params['temperature']}")
        print(f"   - Tools provided: {len(params['tools'])}")
        print(f"   - History included: {'previous context' in params['system']}")

    @patch("anthropic.Anthropic")
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic):
        """Test full two-round sequential tool calling"""
        print("\n🔍 Testing sequential tool calling (two rounds)...")

        # Setup mock client
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Round 1: Claude requests first tool
        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.id = "tool_1"
        mock_tool_content_1.name = "get_course_outline"
        mock_tool_content_1.input = {"course_title": "MCP"}
        mock_response_1.content = [mock_tool_content_1]

        # Round 2: Claude requests second tool
        mock_response_2 = Mock()
        mock_response_2.stop_reason = "tool_use"
        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.id = "tool_2"
        mock_tool_content_2.name = "search_course_content"
        mock_tool_content_2.input = {"query": "lesson 1", "course_name": "MCP"}
        mock_response_2.content = [mock_tool_content_2]

        # Final response after 2 rounds
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Complete answer based on both tools"

        # Setup call sequence
        mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_final_response,
        ]

        # Test
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        mock_tool_manager = MockToolManager()

        response = ai_gen.generate_response(
            "What's in lesson 1 of the MCP course?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
        )

        # Verify
        self.assertEqual(response, "Complete answer based on both tools")
        self.assertEqual(
            mock_client.messages.create.call_count, 3
        )  # 2 tool rounds + final

        print("✅ Sequential tool calling (two rounds) successful")
        print(f"   - API calls made: {mock_client.messages.create.call_count}")
        print(f"   - Final response: {response}")

    @patch("anthropic.Anthropic")
    def test_sequential_tool_calling_early_termination(self, mock_anthropic):
        """Test early termination when Claude provides direct response"""
        print("\n🔍 Testing sequential tool calling early termination...")

        # Setup mock client
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # First response is direct (no tools)
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Direct response without tools"
        mock_client.messages.create.return_value = mock_response

        # Test
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        mock_tool_manager = MockToolManager()

        response = ai_gen.generate_response(
            "What is machine learning?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
        )

        # Verify
        self.assertEqual(response, "Direct response without tools")
        self.assertEqual(mock_client.messages.create.call_count, 1)  # Only first call

        print("✅ Sequential tool calling early termination successful")
        print(f"   - API calls made: {mock_client.messages.create.call_count}")

    @patch("anthropic.Anthropic")
    def test_max_tool_rounds_enforcement(self, mock_anthropic):
        """Test that system stops after max rounds even if Claude wants more tools"""
        print("\n🔍 Testing max tool rounds enforcement...")

        # Setup mock client
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Both rounds request tools
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.id = "tool_123"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_content]

        # Final response without tools
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Forced final response after max rounds"

        # Setup sequence: 2 tool responses + forced final
        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_tool_response,
            mock_final_response,
        ]

        # Test
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        mock_tool_manager = MockToolManager()

        response = ai_gen.generate_response(
            "Complex query requiring multiple searches",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
        )

        # Verify
        self.assertEqual(response, "Forced final response after max rounds")
        self.assertEqual(mock_client.messages.create.call_count, 3)  # 2 rounds + final

        print("✅ Max tool rounds enforcement successful")
        print(f"   - API calls made: {mock_client.messages.create.call_count}")

    @patch("anthropic.Anthropic")
    def test_tool_execution_error_handling_in_sequence(self, mock_anthropic):
        """Test handling of tool execution errors during sequential calling"""
        print("\n🔍 Testing tool execution error handling in sequence...")

        # Setup mock client
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.id = "tool_123"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_content]

        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Response with error handled"

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Test with error-returning tool manager
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        mock_tool_manager = MockToolManager(return_error=True)

        response = ai_gen.generate_response(
            "test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
        )

        # Verify
        self.assertEqual(response, "Response with error handled")
        self.assertEqual(mock_client.messages.create.call_count, 2)

        print("✅ Tool execution error handling successful")

    @patch("anthropic.Anthropic")
    def test_backward_compatibility_single_tool_call(self, mock_anthropic):
        """Test that existing single tool call behavior still works"""
        print("\n🔍 Testing backward compatibility...")

        # Setup mock client for single tool call scenario
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.id = "tool_123"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_content]

        # Direct final response (no more tools)
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Single tool call response"

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Test
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        mock_tool_manager = MockToolManager()

        response = ai_gen.generate_response(
            "search query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            # Default max_tool_rounds=2, but should stop after first round
        )

        # Verify
        self.assertEqual(response, "Single tool call response")
        self.assertEqual(mock_client.messages.create.call_count, 2)  # Tool + final

        print("✅ Backward compatibility maintained")

    @patch("anthropic.Anthropic")
    def test_conversation_context_preserved_across_rounds(self, mock_anthropic):
        """Test that conversation context is maintained across tool rounds"""
        print("\n🔍 Testing conversation context preservation...")

        # Setup mock client
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Tool response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.id = "tool_123"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_content]

        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Response with context"

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Test
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        mock_tool_manager = MockToolManager()

        response = ai_gen.generate_response(
            "follow up question",
            conversation_history="Previous: What is AI?\nAssistant: AI is artificial intelligence.",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
        )

        # Verify that conversation history is included in system prompt
        first_call_args = mock_client.messages.create.call_args_list[0]
        system_prompt = first_call_args.kwargs["system"]
        self.assertIn("Previous conversation", system_prompt)
        self.assertIn("artificial intelligence", system_prompt)

        print("✅ Conversation context preservation successful")


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)
