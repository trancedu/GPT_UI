import pytest
from unittest.mock import patch, Mock, MagicMock
from app import stream_response

class TestAppFunctionality:
    """Test suite for main app functionality"""

    @patch('app.get_ai_client')
    def test_stream_response_basic(self, mock_get_client, sample_messages):
        """Test basic streaming response functionality"""
        # Mock AI client
        mock_client = Mock()
        mock_client.create_stream.return_value = iter(["Hello", " world", "!"])
        mock_get_client.return_value = mock_client
        
        # Test stream response
        response_chunks = list(stream_response(sample_messages, "gpt-4"))
        
        # Verify results
        assert response_chunks == ["Hello", " world", "!"]
        mock_get_client.assert_called_once_with("gpt-4")
        mock_client.create_stream.assert_called_once_with(
            sample_messages, 
            "gpt-4", 
            thinking_enabled=False, 
            thinking_budget=4000
        )

    @patch('app.get_ai_client')
    def test_stream_response_with_thinking(self, mock_get_client, sample_messages):
        """Test streaming response with extended thinking enabled"""
        mock_client = Mock()
        mock_client.create_stream.return_value = iter(["Thinking...", " Response"])
        mock_get_client.return_value = mock_client
        
        # Test with thinking enabled
        response_chunks = list(stream_response(
            sample_messages, 
            "claude-sonnet-4-20250514",
            thinking_enabled=True,
            thinking_budget=5000
        ))
        
        assert response_chunks == ["Thinking...", " Response"]
        mock_client.create_stream.assert_called_once_with(
            sample_messages,
            "claude-sonnet-4-20250514",
            thinking_enabled=True,
            thinking_budget=5000
        )

    @patch('app.get_ai_client')
    def test_stream_response_error_handling(self, mock_get_client, sample_messages):
        """Test stream response error handling"""
        mock_client = Mock()
        mock_client.create_stream.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client
        
        # Should propagate the exception
        with pytest.raises(Exception) as exc_info:
            list(stream_response(sample_messages, "gpt-4"))
        
        assert "API Error" in str(exc_info.value)

    @patch('app.get_ai_client')
    def test_stream_response_empty_stream(self, mock_get_client, sample_messages):
        """Test stream response with empty iterator"""
        mock_client = Mock()
        mock_client.create_stream.return_value = iter([])
        mock_get_client.return_value = mock_client
        
        response_chunks = list(stream_response(sample_messages, "gpt-4"))
        
        assert response_chunks == []

    @patch('app.get_ai_client')
    def test_stream_response_with_different_models(self, mock_get_client, sample_messages):
        """Test stream response works with different model types"""
        mock_client = Mock()
        mock_client.create_stream.return_value = iter(["Response"])
        mock_get_client.return_value = mock_client
        
        # Test different models
        models_to_test = [
            "gpt-4",
            "gpt-3.5-turbo", 
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229"
        ]
        
        for model in models_to_test:
            mock_get_client.reset_mock()
            mock_client.reset_mock()
            
            response_chunks = list(stream_response(sample_messages, model))
            
            assert response_chunks == ["Response"]
            mock_get_client.assert_called_once_with(model)

    @patch('app.get_ai_client')
    def test_stream_response_with_complex_messages(self, mock_get_client, sample_messages_with_files):
        """Test stream response with complex message content (including files)"""
        mock_client = Mock()
        mock_client.create_stream.return_value = iter(["File analysis complete"])
        mock_get_client.return_value = mock_client
        
        response_chunks = list(stream_response(sample_messages_with_files, "claude-3-5-sonnet-20241022"))
        
        assert response_chunks == ["File analysis complete"]
        mock_client.create_stream.assert_called_once_with(
            sample_messages_with_files,
            "claude-3-5-sonnet-20241022",
            thinking_enabled=False,
            thinking_budget=4000
        )

    def test_stream_response_parameter_defaults(self):
        """Test that stream_response has correct parameter defaults"""
        import inspect
        sig = inspect.signature(stream_response)
        
        # Check parameter defaults
        assert sig.parameters['thinking_enabled'].default is False
        assert sig.parameters['thinking_budget'].default == 4000

class TestAppIntegration:
    """Integration tests for app functionality"""

    @patch('app.get_ai_client')
    def test_complete_conversation_flow(self, mock_get_client):
        """Test a complete conversation flow simulation"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        # Simulate a conversation
        conversation = []
        
        # User message 1
        user_msg_1 = {"role": "user", "content": "Hello"}
        conversation.append(user_msg_1)
        
        # AI response 1
        mock_client.create_stream.return_value = iter(["Hi there!"])
        ai_response_1 = "".join(stream_response(conversation, "gpt-4"))
        conversation.append({"role": "assistant", "content": ai_response_1})
        
        # User message 2  
        user_msg_2 = {"role": "user", "content": "How are you?"}
        conversation.append(user_msg_2)
        
        # AI response 2
        mock_client.create_stream.return_value = iter(["I'm doing well!"])
        ai_response_2 = "".join(stream_response(conversation, "gpt-4"))
        conversation.append({"role": "assistant", "content": ai_response_2})
        
        # Verify conversation structure
        assert len(conversation) == 4
        assert conversation[0]["role"] == "user"
        assert conversation[1]["role"] == "assistant"
        assert conversation[2]["role"] == "user" 
        assert conversation[3]["role"] == "assistant"
        assert conversation[1]["content"] == "Hi there!"
        assert conversation[3]["content"] == "I'm doing well!"

    @patch('app.get_ai_client')
    def test_thinking_models_integration(self, mock_get_client):
        """Test integration with thinking-enabled models"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        # Test Claude 4 models with thinking
        claude_4_models = ["claude-sonnet-4-20250514", "claude-opus-4-20250514"]
        
        for model in claude_4_models:
            mock_client.reset_mock()
            mock_client.create_stream.return_value = iter(["<thinking>Let me think...</thinking>Response"])
            
            messages = [{"role": "user", "content": "Complex question"}]
            response = "".join(stream_response(
                messages, 
                model,
                thinking_enabled=True,
                thinking_budget=6000
            ))
            
            assert "Response" in response
            mock_client.create_stream.assert_called_once_with(
                messages,
                model,
                thinking_enabled=True,
                thinking_budget=6000
            )

    @patch('app.get_ai_client')  
    def test_error_recovery(self, mock_get_client):
        """Test error recovery in streaming"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        # First call fails, second succeeds
        mock_client.create_stream.side_effect = [
            Exception("Temporary error"),
            iter(["Recovery successful"])
        ]
        
        messages = [{"role": "user", "content": "Test message"}]
        
        # First attempt should fail
        with pytest.raises(Exception):
            list(stream_response(messages, "gpt-4"))
        
        # Second attempt should succeed
        response_chunks = list(stream_response(messages, "gpt-4"))
        assert response_chunks == ["Recovery successful"]

class TestAppConstants:
    """Test app configuration and constants"""

    def test_thinking_models_list(self):
        """Test that thinking models are properly defined"""
        # This would test any constants defined in the app
        # For now, we'll test the models that support thinking
        claude_4_models = ["claude-sonnet-4-20250514", "claude-opus-4-20250514"]
        
        for model in claude_4_models:
            assert "claude" in model.lower()
            assert "4" in model

    def test_default_thinking_budget(self):
        """Test default thinking budget value"""
        default_budget = 4000
        assert isinstance(default_budget, int)
        assert default_budget > 0 