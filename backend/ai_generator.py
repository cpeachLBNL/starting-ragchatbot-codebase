import openai
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt template for sequential function calling
    SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant specialized in course materials and educational content with access to comprehensive functions for course information.

Available Functions:
1. **search_course_content**: Use for questions about specific course content, lessons, or detailed educational materials
2. **get_course_outline**: Use for questions asking for course outlines, lesson lists, course structure, or "what's in" a course

Multi-Round Function Usage:
- **You can make up to {max_tool_rounds} separate function calls** to gather comprehensive information
- **Sequential reasoning**: Use results from first function call to inform subsequent function calls
- **Strategic approach**: Get overview first (outline), then detailed content (search), or compare multiple sources
- **Synthesize all results**: Combine information from multiple rounds into comprehensive answers

Examples of effective multi-round usage:
- Round 1: Get course outline → Round 2: Search specific lesson content mentioned in outline
- Round 1: Broad topic search → Round 2: Refined search in specific course/lesson based on initial results  
- Round 1: Search course A for topic → Round 2: Search course B for same topic to compare
- Round 1: Get course outline to find relevant lessons → Round 2: Search those specific lessons

Function Selection Guidelines:
- **Course outline queries** (outline, syllabus, lesson list, course structure): Use get_course_outline function
- **Content queries** (specific topics, lesson details, explanations): Use search_course_content function
- **Comparison queries**: Use multiple searches or outlines as needed
- **Complex questions**: Break down into multiple function calls for thorough coverage

For Course Outline Responses:
- Always include the course title, instructor, and course link when available
- List all lessons with their numbers and titles
- Present information clearly and completely from the function results

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using functions
- **Course-specific questions**: Use appropriate functions across multiple rounds if needed
- **Complex queries**: Don't hesitate to use multiple function calls for comprehensive answers
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results", "using functions", or "in my first/second search"

All responses must be:
1. **Comprehensive** - Include all relevant information gathered from function results
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language  
4. **Well-synthesized** - Combine multiple function results seamlessly
5. **Example-supported** - Include relevant examples when they aid understanding

Provide only the direct answer to what was asked, incorporating all gathered information.

{conversation_context}"""
    
    def __init__(self, api_key: str, model: str, base_url: str = None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_tool_rounds: int = 2) -> str:
        """
        Generate AI response with support for sequential function calling.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available functions the AI can use
            tool_manager: Manager to execute functions
            max_tool_rounds: Maximum number of function execution rounds (default 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content with function round information
        system_content = self._build_system_content(conversation_history, max_tool_rounds)
        
        # Initialize conversation with system and user messages
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
        
        # Sequential function execution loop
        for round_num in range(1, max_tool_rounds + 1):
            # Make API call with functions available
            response = self._make_api_call(messages, tools)
            
            # Check if tools are requested
            if response.choices[0].finish_reason != "tool_calls":
                # GPT provided direct response, no tools needed
                return response.choices[0].message.content
            
            # Verify function manager is available
            if not tool_manager:
                return "Function use requested but no tool manager available"
            
            # Add GPT's response (with tool calls) to conversation
            messages.append({
                "role": "assistant", 
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls
            })
            
            # Execute tools and add results to conversation
            tool_results = self._execute_tools(response, tool_manager)
            messages.extend(tool_results)
        
        # After max rounds, make final API call without tools
        final_response = self._make_api_call(messages, functions=None)
        return final_response.choices[0].message.content
    
    def _build_system_content(self, conversation_history: Optional[str], max_tool_rounds: int) -> str:
        """Build system prompt with conversation history and tool round information"""
        conversation_context = (
            f"\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history else ""
        )
        
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            max_tool_rounds=max_tool_rounds,
            conversation_context=conversation_context
        )
    
    def _make_api_call(self, messages: List[Dict[str, Any]], 
                       functions: Optional[List] = None):
        """Make single API call with error handling"""
        api_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Add tools if provided (OpenAI now uses tools instead of functions)
        if functions:
            # Convert function definitions to tool format
            tools = []
            for func in functions:
                tools.append({
                    "type": "function",
                    "function": func
                })
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"
        
        try:
            return self.client.chat.completions.create(**api_params)
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")
    
    def _execute_tools(self, response, tool_manager) -> List[Dict[str, Any]]:
        """Execute tool calls from response and return formatted results"""
        tool_results = []
        
        for tool_call in response.choices[0].message.tool_calls:
            try:
                import json
                # Parse tool arguments
                arguments = json.loads(tool_call.function.arguments)
                
                # Execute the tool
                result = tool_manager.execute_tool(
                    tool_call.function.name,
                    **arguments
                )
                
                # Add tool result message
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            except Exception as e:
                # Handle tool execution errors gracefully
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Tool execution error: {str(e)}"
                })
        
        return tool_results
    
