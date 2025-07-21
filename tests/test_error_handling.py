import pytest
import os
import time
from unittest.mock import patch, Mock, MagicMock
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_client import ClaudeClient


class TestErrorHandling:
    """Test suite for error handling and retry logic"""

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_overloaded_error_handling(self, mock_anthropic, mock_env_get):
        """Test handling of overloaded API errors"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock overloaded error
        mock_client.messages.create.side_effect = Exception("overloaded_error")
        
        claude_client = ClaudeClient()
        
        messages = [{"role": "user", "content": "Test message"}]
        response = claude_client.create_response(messages, "claude-sonnet-4-20250514")
        
        # Should return overloaded error message
        assert "❌ **API Overloaded**" in response
        assert "Please try again in a few minutes" in response

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_rate_limit_error_handling(self, mock_anthropic, mock_env_get):
        """Test handling of rate limit errors"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock rate limit error
        mock_client.messages.create.side_effect = Exception("rate_limit")
        
        claude_client = ClaudeClient()
        
        messages = [{"role": "user", "content": "Test message"}]
        response = claude_client.create_response(messages, "claude-sonnet-4-20250514")
        
        # Should return rate limit error message
        assert "❌ **Rate Limited**" in response
        assert "Please wait a moment" in response

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_authentication_error_handling(self, mock_anthropic, mock_env_get):
        """Test handling of authentication errors"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock authentication error
        mock_client.messages.create.side_effect = Exception("401 authentication")
        
        claude_client = ClaudeClient()
        
        messages = [{"role": "user", "content": "Test message"}]
        response = claude_client.create_response(messages, "claude-sonnet-4-20250514")
        
        # Should return authentication error message
        assert "❌ **Authentication Error**" in response
        assert "check your API key" in response

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_web_search_error_handling(self, mock_anthropic, mock_env_get):
        """Test handling of web search specific errors"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock web search error
        mock_client.messages.create.side_effect = Exception("web_search unavailable")
        
        claude_client = ClaudeClient()
        
        messages = [{"role": "user", "content": "Test message"}]
        response = claude_client.create_response(messages, "claude-sonnet-4-20250514")
        
        # Should return web search error message
        assert "❌ **Web Search Error**" in response
        assert "currently unavailable" in response

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_generic_error_handling(self, mock_anthropic, mock_env_get):
        """Test handling of generic errors"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock generic error
        mock_client.messages.create.side_effect = Exception("Something went wrong")
        
        claude_client = ClaudeClient()
        
        messages = [{"role": "user", "content": "Test message"}]
        response = claude_client.create_response(messages, "claude-sonnet-4-20250514")
        
        # Should return generic error message
        assert "❌ **Error**" in response
        assert "Something went wrong" in response

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_retry_logic_for_overloaded_errors(self, mock_anthropic, mock_env_get):
        """Test that overloaded errors trigger retry logic"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock successful response
        mock_response = Mock()
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Success!"
        mock_response.content = [mock_text_block]
        
        # Mock overloaded error for first 2 calls, then success
        mock_client.messages.create.side_effect = [
            Exception("overloaded_error"),
            Exception("overloaded_error"),
            mock_response  # Success on third try
        ]
        
        claude_client = ClaudeClient()
        
        messages = [{"role": "user", "content": "Test message"}]
        response = claude_client.create_response(messages, "claude-sonnet-4-20250514")
        
        # Should eventually succeed
        assert "Success!" in response
        # Should have been called 3 times (2 failures + 1 success)
        assert mock_client.messages.create.call_count == 3

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_max_retries_exceeded(self, mock_anthropic, mock_env_get):
        """Test that max retries are respected"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock overloaded error for all attempts
        mock_client.messages.create.side_effect = Exception("overloaded_error")
        
        claude_client = ClaudeClient()
        
        messages = [{"role": "user", "content": "Test message"}]
        response = claude_client.create_response(messages, "claude-sonnet-4-20250514")
        
        # Should return final error message after max retries
        assert "❌ **API Overloaded**" in response
        assert "Please try again in a few minutes" in response
        # Should have been called 4 times (initial + 3 retries)
        assert mock_client.messages.create.call_count == 4

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_streaming_error_handling(self, mock_anthropic, mock_env_get):
        """Test error handling in streaming mode"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock overloaded error
        mock_client.messages.create.side_effect = Exception("overloaded_error")
        
        claude_client = ClaudeClient()
        
        messages = [{"role": "user", "content": "Test message"}]
        stream = claude_client.create_stream(messages, "claude-sonnet-4-20250514")
        
        # Collect stream results
        results = list(stream)
        
        # Should return overloaded error message
        result_text = "".join(results)
        assert "❌ **API Overloaded**" in result_text
        assert "Please try again in a few minutes" in result_text

    @patch('ai_client.os.environ.get')
    @patch('anthropic.Anthropic')
    def test_retry_with_exponential_backoff(self, mock_anthropic, mock_env_get):
        """Test that retries use exponential backoff"""
        mock_env_get.return_value = "test-api-key"
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock successful response
        mock_response = Mock()
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Success!"
        mock_response.content = [mock_text_block]
        
        # Mock overloaded error for first call, then success
        mock_client.messages.create.side_effect = [
            Exception("overloaded_error"),
            mock_response  # Success on second try
        ]
        
        claude_client = ClaudeClient()
        
        start_time = time.time()
        messages = [{"role": "user", "content": "Test message"}]
        response = claude_client.create_response(messages, "claude-sonnet-4-20250514")
        end_time = time.time()
        
        # Should have taken at least 1 second (exponential backoff)
        assert end_time - start_time >= 1.0
        assert "Success!" in response

    def test_error_message_formatting(self):
        """Test that error messages are properly formatted"""
        client = ClaudeClient.__new__(ClaudeClient)  # Create without __init__
        
        # Test overloaded error
        error = Exception("overloaded_error")
        message = client._handle_api_error(error, 0, 3)
        assert "🔄 **API Overloaded**" in message
        assert "Retrying" in message
        
        # Test final overloaded error
        message = client._handle_api_error(error, 3, 3)
        assert "❌ **API Overloaded**" in message
        assert "Please try again in a few minutes" in message
        
        # Test rate limit error
        error = Exception("rate_limit")
        message = client._handle_api_error(error, 0, 3)
        assert "🔄 **Rate Limited**" in message
        
        # Test authentication error
        error = Exception("401 authentication")
        message = client._handle_api_error(error, 0, 3)
        assert "❌ **Authentication Error**" in message
        assert "check your API key" in message
        
        # Test web search error
        error = Exception("web_search unavailable")
        message = client._handle_api_error(error, 0, 3)
        assert "❌ **Web Search Error**" in message
        assert "currently unavailable" in message
        
        # Test generic error
        error = Exception("Something went wrong")
        message = client._handle_api_error(error, 0, 3)
        assert "❌ **Error**" in message
        assert "Something went wrong" in message 