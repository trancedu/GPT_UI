import pytest
from unittest.mock import patch, Mock
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports_work():
    """Test that we can import our modules"""
    # These should not raise ImportError
    from chat_history import ensure_chat_directory
    from ai_client import get_ai_client
    from app import stream_response
    import file_manager
    
    assert True  # If we get here, imports worked

def test_file_manager_format_file_size():
    """Test a simple utility function"""
    from file_manager import format_file_size
    
    # Test bytes
    assert format_file_size(100) == "100B"
    
    # Test KB
    assert format_file_size(1024) == "1.0KB" 
    assert format_file_size(1536) == "1.5KB"
    
    # Test MB
    assert format_file_size(1048576) == "1.0MB"
    assert format_file_size(2097152) == "2.0MB"

def test_directory_creation():
    """Test directory creation functionality"""
    import tempfile
    import shutil
    from chat_history import ensure_chat_directory
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    test_dir = os.path.join(temp_dir, "test_chats")
    
    try:
        # Mock the CHAT_HISTORY_DIR to use our test directory
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            # Directory should not exist initially
            assert not os.path.exists(test_dir)
            
            # Call ensure_chat_directory
            ensure_chat_directory()
            
            # Directory should now exist
            assert os.path.exists(test_dir)
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

@patch.dict(os.environ, {}, clear=True)  
def test_environment_mocking():
    """Test that environment variable mocking works"""
    # Environment should be empty due to patch
    assert 'OPENAI_API_KEY' not in os.environ
    assert 'ANTHROPIC_API_KEY' not in os.environ
    
    # Now patch in test values
    with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
        assert os.environ.get('TEST_VAR') == 'test_value'
    
    # Should be gone after context
    assert 'TEST_VAR' not in os.environ

def test_mock_functionality():
    """Test that mock functionality works properly"""
    mock_obj = Mock()
    mock_obj.test_method.return_value = "mocked_result"
    
    result = mock_obj.test_method()
    assert result == "mocked_result"
    
    # Check call was recorded
    mock_obj.test_method.assert_called_once()

class TestBasicFunctionality:
    """Test basic class-based testing works"""
    
    def test_simple_assertion(self):
        """Simple test that should always pass"""
        assert 2 + 2 == 4
        
    def test_list_operations(self):
        """Test basic list operations"""
        test_list = [1, 2, 3]
        assert len(test_list) == 3
        assert 2 in test_list
        
    def test_string_operations(self):
        """Test basic string operations"""
        test_string = "Hello, World!"
        assert test_string.startswith("Hello")
        assert test_string.endswith("!")
        assert "World" in test_string 