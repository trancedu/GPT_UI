import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, Mock, MagicMock
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_client import ClaudeClient, get_ai_client


class TestWebSearchTool:
    """Test suite for web search tool functionality"""

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_create_web_search_tool_basic(self, mock_anthropic, mock_env_get):
        """Test creating basic web search tool"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        claude_client = ClaudeClient()
        
        # Test basic web search tool creation
        tool = claude_client.create_web_search_tool()
        
        expected = {
            "type": "web_search_20250305",
            "name": "web_search"
        }
        assert tool == expected

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_create_web_search_tool_with_parameters(self, mock_anthropic, mock_env_get):
        """Test creating web search tool with all parameters"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        claude_client = ClaudeClient()
        
        # Test web search tool with parameters
        tool = claude_client.create_web_search_tool(
            max_uses=5,
            allowed_domains=["example.com", "test.org"],
            blocked_domains=["spam.com", "ads.net"],
            user_location="New York, US"
        )
        
        expected = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5,
            "allowed_domains": ["example.com", "test.org"],
            "blocked_domains": ["spam.com", "ads.net"],
            "user_location": "New York, US"
        }
        assert tool == expected

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_create_web_search_tool_partial_parameters(self, mock_anthropic, mock_env_get):
        """Test creating web search tool with some parameters"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        claude_client = ClaudeClient()
        
        # Test web search tool with only some parameters
        tool = claude_client.create_web_search_tool(
            max_uses=3,
            user_location="London, UK"
        )
        
        expected = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 3,
            "user_location": "London, UK"
        }
        assert tool == expected

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_create_stream_with_web_search_tool(self, mock_anthropic, mock_env_get):
        """Test streaming with web search tool"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock streaming response with web search
        mock_event_1 = Mock()
        mock_event_1.type = "content_block_start"
        mock_event_1.content_block.type = "tool_use"
        mock_event_1.content_block.name = "web_search"
        
        mock_event_2 = Mock()
        mock_event_2.type = "content_block_delta"
        mock_delta = Mock()
        mock_delta.text = "Based on my web search..."
        # Add hasattr support for the actual code checks
        mock_delta.thinking = None  # So hasattr(event.delta, 'thinking') is False
        mock_event_2.delta = mock_delta
        
        mock_event_3 = Mock()
        mock_event_3.type = "message_stop"
        
        mock_client.messages.create.return_value = iter([mock_event_1, mock_event_2, mock_event_3])
        
        claude_client = ClaudeClient()
        
        # Create web search tool
        web_search_tool = claude_client.create_web_search_tool(max_uses=3)
        tools = [web_search_tool]
        
        # Test streaming with tools
        messages = [{"role": "user", "content": "What's the latest news?"}]
        stream = claude_client.create_stream(messages, "claude-sonnet-4-20250514", tools=tools)
        
        # Collect stream results
        results = list(stream)
        
        # Verify web search indicator and response
        assert any("🔍 **Searching the web:**" in result for result in results)
        assert any("Based on my web search..." in result for result in results)

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_create_stream_with_web_search_beta_header(self, mock_anthropic, mock_env_get):
        """Test that web search adds correct beta header"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock simple response
        mock_client.messages.create.return_value = iter([])
        
        claude_client = ClaudeClient()
        
        # Create web search tool
        web_search_tool = claude_client.create_web_search_tool()
        tools = [web_search_tool]
        
        messages = [{"role": "user", "content": "Search the web"}]
        
        # Call create_stream
        list(claude_client.create_stream(messages, "claude-sonnet-4-20250514", tools=tools))
        
        # Verify the client was called with correct beta header
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        
        assert "extra_headers" in call_args.kwargs
        assert "anthropic-beta" in call_args.kwargs["extra_headers"]
        assert "web-search-2025-03-05" in call_args.kwargs["extra_headers"]["anthropic-beta"]

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_create_stream_with_files_and_web_search_headers(self, mock_anthropic, mock_env_get):
        """Test that both file and web search beta headers are combined"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock simple response
        mock_client.messages.create.return_value = iter([])
        
        claude_client = ClaudeClient()
        
        # Create web search tool
        web_search_tool = claude_client.create_web_search_tool()
        tools = [web_search_tool]
        
        # Create messages with file reference
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Analyze this file and search for updates"},
                    {
                        "type": "document",
                        "source": {
                            "type": "file",
                            "file_id": "file_123"
                        }
                    }
                ]
            }
        ]
        
        # Call create_stream
        list(claude_client.create_stream(messages, "claude-sonnet-4-20250514", tools=tools))
        
        # Verify the client was called with combined beta headers
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        
        assert "extra_headers" in call_args.kwargs
        beta_header = call_args.kwargs["extra_headers"]["anthropic-beta"]
        assert "files-api-2025-04-14" in beta_header
        assert "web-search-2025-03-05" in beta_header

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_create_response_with_web_search_tool(self, mock_anthropic, mock_env_get):
        """Test non-streaming response with web search tool"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock response with tool use
        mock_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "web_search"
        mock_tool_block.input = {"query": "latest AI news"}
        
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Based on the search results, here's what I found..."
        
        mock_response.content = [mock_tool_block, mock_text_block]
        mock_client.messages.create.return_value = mock_response
        
        claude_client = ClaudeClient()
        
        # Create web search tool
        web_search_tool = claude_client.create_web_search_tool(max_uses=5)
        tools = [web_search_tool]
        
        messages = [{"role": "user", "content": "What are the latest AI developments?"}]
        response = claude_client.create_response(messages, "claude-sonnet-4-20250514", tools=tools)
        
        # Verify response includes web search indication and content
        assert "🔍 **Web search performed:** latest AI news" in response
        assert "Based on the search results, here's what I found..." in response

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_create_response_without_web_search(self, mock_anthropic, mock_env_get):
        """Test that response works normally without web search tools"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock normal response
        mock_response = Mock()
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "I can help you with that question."
        
        mock_response.content = [mock_text_block]
        mock_client.messages.create.return_value = mock_response
        
        claude_client = ClaudeClient()
        
        messages = [{"role": "user", "content": "Hello"}]
        response = claude_client.create_response(messages, "claude-sonnet-4-20250514")
        
        # Verify normal response without web search indicators
        assert response == "I can help you with that question."
        assert "🔍" not in response

    def test_web_search_tool_validation(self):
        """Test web search tool parameter validation"""
        # Test that we can create tools with valid parameters
        client = ClaudeClient.__new__(ClaudeClient)  # Create without __init__
        
        # Test max_uses validation (should accept positive integers)
        tool = client.create_web_search_tool(max_uses=1)
        assert tool["max_uses"] == 1
        
        tool = client.create_web_search_tool(max_uses=10)
        assert tool["max_uses"] == 10
        
        # Test that empty lists work
        tool = client.create_web_search_tool(allowed_domains=[], blocked_domains=[])
        # Empty lists should be treated as None (not included)
        assert "allowed_domains" not in tool
        assert "blocked_domains" not in tool
        
        # Test that None values are not included
        tool = client.create_web_search_tool(
            max_uses=None,
            allowed_domains=None,
            blocked_domains=None,
            user_location=None
        )
        expected_keys = {"type", "name"}
        assert set(tool.keys()) == expected_keys


class TestWebSearchIntegration:
    """Integration tests for web search functionality"""

    @patch('ai_client.os.environ.get')
    def test_get_ai_client_supports_web_search(self, mock_env_get):
        """Test that getting Claude client supports web search"""
        mock_env_get.return_value = "test-api-key"
        
        with patch('anthropic.Anthropic'):
            client = get_ai_client("claude-sonnet-4-20250514")
            
            # Verify client has web search method
            assert hasattr(client, 'create_web_search_tool')
            assert callable(client.create_web_search_tool)

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_web_search_with_thinking_integration(self, mock_anthropic, mock_env_get):
        """Test web search combined with thinking"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock response with thinking and web search
        mock_thinking_event = Mock()
        mock_thinking_event.type = "content_block_start"
        mock_thinking_event.content_block.type = "thinking"
        
        mock_thinking_delta = Mock()
        mock_thinking_delta.type = "content_block_delta"
        mock_thinking_delta_obj = Mock()
        mock_thinking_delta_obj.thinking = "I need to search for recent information..."
        mock_thinking_delta_obj.text = None  # So hasattr(..., 'text') returns False
        mock_thinking_delta.delta = mock_thinking_delta_obj
        
        mock_tool_event = Mock()
        mock_tool_event.type = "content_block_start"
        mock_tool_event.content_block.type = "tool_use"
        mock_tool_event.content_block.name = "web_search"
        
        mock_text_delta = Mock()
        mock_text_delta.type = "content_block_delta"
        mock_text_delta_obj = Mock()
        mock_text_delta_obj.text = "Based on my search..."
        mock_text_delta_obj.thinking = None  # So hasattr(..., 'thinking') returns False
        mock_text_delta.delta = mock_text_delta_obj
        
        mock_stop = Mock()
        mock_stop.type = "message_stop"
        
        mock_client.messages.create.return_value = iter([
            mock_thinking_event, mock_thinking_delta, mock_tool_event, 
            mock_text_delta, mock_stop
        ])
        
        claude_client = ClaudeClient()
        
        # Create web search tool
        web_search_tool = claude_client.create_web_search_tool()
        tools = [web_search_tool]
        
        messages = [{"role": "user", "content": "What's happening in AI today?"}]
        stream = claude_client.create_stream(
            messages, 
            "claude-sonnet-4-20250514", 
            thinking_enabled=True,
            tools=tools
        )
        
        results = list(stream)
        
        # Verify both thinking and web search indicators
        result_text = "".join(results)
        assert "🧠 **Claude is thinking:**" in result_text
        assert "I need to search for recent information..." in result_text
        assert "🔍 **Searching the web:**" in result_text
        assert "Based on my search..." in result_text

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_web_search_error_handling(self, mock_anthropic, mock_env_get):
        """Test error handling in web search"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock API error
        mock_client.messages.create.side_effect = Exception("Web search API error")
        
        claude_client = ClaudeClient()
        
        # Create web search tool
        web_search_tool = claude_client.create_web_search_tool()
        tools = [web_search_tool]
        
        messages = [{"role": "user", "content": "Search for something"}]
        stream = claude_client.create_stream(messages, "claude-sonnet-4-20250514", tools=tools)
        
        results = list(stream)
        
        # Verify error is handled gracefully
        assert len(results) == 1
        assert "❌ **Error**: Web search API error" in results[0]


class TestWebSearchUISupport:
    """Tests for web search UI configuration support functions"""
    
    def test_domain_parsing(self):
        """Test domain parsing logic"""
        # This tests the domain parsing logic that would be used in the UI
        
        # Test comma-separated domains
        domains_text = "example.com, test.org, news.site.com"
        parsed = [d.strip() for d in domains_text.replace(',', '\n').split('\n') if d.strip()]
        expected = ["example.com", "test.org", "news.site.com"]
        assert parsed == expected
        
        # Test newline-separated domains
        domains_text = "example.com\ntest.org\nnews.site.com"
        parsed = [d.strip() for d in domains_text.replace(',', '\n').split('\n') if d.strip()]
        expected = ["example.com", "test.org", "news.site.com"]
        assert parsed == expected
        
        # Test mixed format
        domains_text = "example.com, test.org\nnews.site.com\n,  another.com  "
        parsed = [d.strip() for d in domains_text.replace(',', '\n').split('\n') if d.strip()]
        expected = ["example.com", "test.org", "news.site.com", "another.com"]
        assert parsed == expected
        
        # Test empty string
        domains_text = ""
        parsed = [d.strip() for d in domains_text.replace(',', '\n').split('\n') if d.strip()]
        assert parsed == []
        
        # Test whitespace only
        domains_text = "   \n  ,  \n  "
        parsed = [d.strip() for d in domains_text.replace(',', '\n').split('\n') if d.strip()]
        assert parsed == []

    def test_web_search_configuration_validation(self):
        """Test web search configuration validation"""
        client = ClaudeClient.__new__(ClaudeClient)  # Create without __init__
        
        # Test valid configurations
        configs = [
            {"max_uses": 1},
            {"max_uses": 10},
            {"allowed_domains": ["example.com"]},
            {"blocked_domains": ["spam.com"]},
            {"user_location": "New York, US"},
            {
                "max_uses": 5,
                "allowed_domains": ["news.com", "tech.org"],
                "blocked_domains": ["ads.com"],
                "user_location": "London, UK"
            }
        ]
        
        for config in configs:
            tool = client.create_web_search_tool(**config)
            assert tool["type"] == "web_search_20250305"
            assert tool["name"] == "web_search"
            
            for key, value in config.items():
                if value is not None:
                    assert tool[key] == value 