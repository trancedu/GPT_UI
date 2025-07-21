import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
import sys

# Add the dev directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def temp_chat_directory():
    """Create a temporary directory for chat history tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_streamlit():
    """Mock streamlit session state for testing"""
    with patch('streamlit.session_state') as mock_st:
        mock_st.messages = []
        mock_st.current_chat_name = None
        mock_st.thinking_enabled = False
        mock_st.thinking_budget = 4000
        mock_st.uploaded_files = []
        mock_st.pending_files = []
        mock_st.last_uploaded_file = None
        yield mock_st

@pytest.fixture
def sample_messages():
    """Sample chat messages for testing"""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
        {"role": "user", "content": "Can you explain machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."}
    ]

@pytest.fixture
def sample_messages_with_files():
    """Sample chat messages with file content for testing"""
    return [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "Analyze this document"},
                {"type": "document", "source": {"type": "file", "file_id": "file_123"}}
            ]
        },
        {"role": "assistant", "content": "I can see the document you uploaded..."}
    ]

@pytest.fixture
def mock_ai_client():
    """Mock AI client for testing"""
    client = Mock()
    client.create_stream.return_value = iter(["Hello", " there", "!"])
    client.create_response.return_value = "Hello there!"
    client.get_available_models.return_value = ["gpt-4", "gpt-3.5-turbo"]
    return client

@pytest.fixture
def mock_openai():
    """Mock OpenAI client"""
    with patch('openai.OpenAI') as mock:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value.choices = [
            Mock(delta=Mock(content="Test response"))
        ]
        mock.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client"""
    with patch('anthropic.Anthropic') as mock:
        mock_client = Mock()
        mock_client.messages.stream.return_value = [
            Mock(type="text", text="Test response")
        ]
        mock.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key'
    }):
        yield 