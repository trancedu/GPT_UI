import pytest
import os
from unittest.mock import patch, Mock, MagicMock
from ai_client import (
    get_ai_client,
    get_available_models,
    get_openai_client,
    OpenAIClient,
    ClaudeClient
)

class TestAIClient:
    """Test suite for AI client functionality"""

    def test_get_available_models(self, mock_env_vars):
        """Test getting available models returns correct structure"""
        models = get_available_models()
        
        assert isinstance(models, dict)
        assert "OpenAI" in models
        assert "Anthropic" in models
        
        # Check OpenAI models
        openai_models = models["OpenAI"]
        assert "gpt-4o" in openai_models
        assert "gpt-4" in openai_models
        assert "gpt-3.5-turbo" in openai_models
        
        # Check Anthropic models
        anthropic_models = models["Anthropic"]
        assert "claude-3-5-sonnet-20241022" in anthropic_models
        assert "claude-3-opus-20240229" in anthropic_models

    def test_get_ai_client_openai(self, mock_env_vars):
        """Test getting OpenAI client for OpenAI models"""
        with patch('ai_client.OpenAIClient') as mock_client:
            client = get_ai_client("gpt-4")
            mock_client.assert_called_once()
            assert client is not None

    def test_get_ai_client_claude(self, mock_env_vars):
        """Test getting Claude client for Claude models"""
        with patch('ai_client.ClaudeClient') as mock_client:
            client = get_ai_client("claude-3-5-sonnet-20241022")
            mock_client.assert_called_once()
            assert client is not None

    def test_get_ai_client_unknown_model(self, mock_env_vars):
        """Test getting client for unknown model defaults to OpenAI"""
        with patch('ai_client.OpenAIClient') as mock_client:
            client = get_ai_client("unknown-model")
            mock_client.assert_called_once()
            assert client is not None

    @patch('openai.OpenAI')
    def test_get_openai_client(self, mock_openai, mock_env_vars):
        """Test getting OpenAI client with API key"""
        client = get_openai_client()
        mock_openai.assert_called_once_with(api_key='test-openai-key')
        assert client is not None

    def test_get_openai_client_no_api_key(self):
        """Test getting OpenAI client without API key raises exception"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception) as exc_info:
                get_openai_client()
            assert "OPENAI_API_KEY" in str(exc_info.value)

class TestOpenAIClient:
    """Test suite for OpenAI client implementation"""

    @patch('ai_client.get_openai_client')
    def test_openai_client_initialization(self, mock_get_client, mock_env_vars):
        """Test OpenAI client initializes correctly"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        openai_client = OpenAIClient()
        assert openai_client.client == mock_client

    @patch('ai_client.get_openai_client')
    def test_create_stream_basic(self, mock_get_client, sample_messages):
        """Test creating stream response"""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_chunk = Mock()
        mock_chunk.choices = [Mock(delta=Mock(content="Test response"))]
        mock_response.__iter__ = Mock(return_value=iter([mock_chunk]))
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        openai_client = OpenAIClient()
        stream = openai_client.create_stream(sample_messages, "gpt-4")
        
        # Collect stream results
        results = list(stream)
        assert len(results) == 1
        assert results[0] == "Test response"

    @patch('ai_client.get_openai_client')
    def test_create_response_basic(self, mock_get_client, sample_messages):
        """Test creating complete response"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Complete response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        openai_client = OpenAIClient()
        response = openai_client.create_response(sample_messages, "gpt-4")
        
        assert response == "Complete response"

    @patch('ai_client.get_openai_client')
    def test_get_available_models_openai(self, mock_get_client):
        """Test getting available OpenAI models"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        openai_client = OpenAIClient()
        models = openai_client.get_available_models()
        
        expected_models = [
            "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo",
            "gpt-3.5-turbo", "o1-preview", "o1-mini"
        ]
        assert models == expected_models

class TestClaudeClient:
    """Test suite for Claude client implementation"""

    @patch('anthropic.Anthropic')
    def test_claude_client_initialization(self, mock_anthropic, mock_env_vars):
        """Test Claude client initializes correctly"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        claude_client = ClaudeClient()
        mock_anthropic.assert_called_once_with(api_key='test-anthropic-key')
        assert claude_client.client == mock_client

    def test_claude_client_no_api_key(self):
        """Test Claude client without API key raises exception"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception) as exc_info:
                ClaudeClient()
            assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    @patch('anthropic.Anthropic')
    def test_create_stream_basic(self, mock_anthropic, sample_messages, mock_env_vars):
        """Test creating stream response with Claude"""
        mock_client = Mock()
        mock_stream = Mock()
        
        # Mock stream events
        mock_text_event = Mock()
        mock_text_event.type = "text"
        mock_text_event.text = "Claude response"
        
        mock_stream.__iter__ = Mock(return_value=iter([mock_text_event]))
        mock_client.messages.stream.return_value.__enter__ = Mock(return_value=mock_stream)
        mock_client.messages.stream.return_value.__exit__ = Mock(return_value=None)
        mock_anthropic.return_value = mock_client
        
        claude_client = ClaudeClient()
        stream = claude_client.create_stream(sample_messages, "claude-3-5-sonnet-20241022")
        
        results = list(stream)
        assert len(results) == 1
        assert results[0] == "Claude response"

    @patch('anthropic.Anthropic')
    def test_create_response_basic(self, mock_anthropic, sample_messages, mock_env_vars):
        """Test creating complete response with Claude"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Complete Claude response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        claude_client = ClaudeClient()
        response = claude_client.create_response(sample_messages, "claude-3-5-sonnet-20241022")
        
        assert response == "Complete Claude response"

    @patch('anthropic.Anthropic')
    def test_get_available_models_claude(self, mock_anthropic, mock_env_vars):
        """Test getting available Claude models"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        claude_client = ClaudeClient()
        models = claude_client.get_available_models()
        
        expected_models = [
            "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307", "claude-sonnet-4-20250514",
            "claude-opus-4-20250514"
        ]
        assert models == expected_models

    @patch('anthropic.Anthropic')  
    def test_thinking_enabled_parameters(self, mock_anthropic, sample_messages, mock_env_vars):
        """Test that thinking parameters are passed correctly for Claude 4 models"""
        mock_client = Mock()
        mock_stream = Mock()
        mock_stream.__iter__ = Mock(return_value=iter([]))
        mock_client.messages.stream.return_value.__enter__ = Mock(return_value=mock_stream)
        mock_client.messages.stream.return_value.__exit__ = Mock(return_value=None)
        mock_anthropic.return_value = mock_client
        
        claude_client = ClaudeClient()
        list(claude_client.create_stream(
            sample_messages, 
            "claude-sonnet-4-20250514",
            thinking_enabled=True,
            thinking_budget=5000
        ))
        
        # Check that the correct parameters were passed
        call_args = mock_client.messages.stream.call_args
        assert call_args.kwargs['thinking']['enabled'] is True
        assert call_args.kwargs['thinking']['budget'] == 5000

class TestIntegration:
    """Integration tests for AI client functionality"""

    def test_model_client_mapping(self, mock_env_vars):
        """Test that different models get mapped to correct clients"""
        test_cases = [
            ("gpt-4", "OpenAI"),
            ("gpt-3.5-turbo", "OpenAI"), 
            ("claude-3-5-sonnet-20241022", "Claude"),
            ("claude-3-opus-20240229", "Claude"),
            ("unknown-model", "OpenAI")  # Default fallback
        ]
        
        for model, expected_provider in test_cases:
            with patch('ai_client.OpenAIClient') as mock_openai, \
                 patch('ai_client.ClaudeClient') as mock_claude:
                
                client = get_ai_client(model)
                
                if expected_provider == "OpenAI":
                    mock_openai.assert_called_once()
                    mock_claude.assert_not_called()
                else:
                    mock_claude.assert_called_once()
                    mock_openai.assert_not_called()
                
                # Reset mocks for next iteration
                mock_openai.reset_mock()
                mock_claude.reset_mock()

    def test_error_handling(self, mock_env_vars):
        """Test error handling in AI clients"""
        with patch('ai_client.OpenAIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.create_response.side_effect = Exception("API Error")
            mock_client_class.return_value = mock_client
            
            client = get_ai_client("gpt-4")
            
            with pytest.raises(Exception) as exc_info:
                client.create_response([], "gpt-4")
            assert "API Error" in str(exc_info.value) 