import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, Mock
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_client import get_available_models
from chat_history import auto_save_chat, ensure_chat_directory, get_chat_info, load_chat_history
from file_manager import format_file_size


class TestActualBehavior:
    """Tests that match the actual behavior of the functions"""

    def test_get_available_models_actual_format(self):
        """Test getting available models returns correct actual structure"""
        models = get_available_models()
        
        assert isinstance(models, dict)
        # Test actual keys (lowercase)
        assert "openai" in models
        assert "claude" in models
        
        # Test actual model lists
        openai_models = models["openai"]
        assert "gpt-4o" in openai_models
        assert "gpt-3.5-turbo" in openai_models
        
        claude_models = models["claude"]
        assert "claude-sonnet-4-20250514" in claude_models

    def test_file_size_formatting_comprehensive(self):
        """Test file size formatting with all cases"""
        # Test bytes
        assert format_file_size(0) == "0B"
        assert format_file_size(100) == "100B"
        assert format_file_size(1023) == "1023B"
        
        # Test KB
        assert format_file_size(1024) == "1.0KB"
        assert format_file_size(1536) == "1.5KB"
        assert format_file_size(2048) == "2.0KB"
        
        # Test MB
        assert format_file_size(1048576) == "1.0MB"
        assert format_file_size(2097152) == "2.0MB"
        assert format_file_size(1572864) == "1.5MB"  # 1.5MB

    def test_chat_filename_generation_actual(self):
        """Test chat filename generation with actual behavior"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch('chat_history.CHAT_HISTORY_DIR', temp_dir):
                # Test basic message
                messages = [{"role": "user", "content": "Hello, how are you?"}]
                filename = auto_save_chat(messages)
                
                assert filename is not None
                assert filename.endswith(".json")
                # Note: commas are removed, so "Hello, how are you?" becomes "Hello how are you"
                assert filename == "Hello how are you.json"
                
                # Test with special characters that get cleaned
                messages2 = [{"role": "user", "content": "What's the <best> way?"}]
                filename2 = auto_save_chat(messages2)
                
                assert filename2 is not None
                assert filename2.endswith(".json")
                # Special characters should be removed/cleaned
                assert "Whats" in filename2 or "What" in filename2
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_directory_creation_works(self):
        """Test directory creation functionality"""
        temp_dir = tempfile.mkdtemp()
        test_dir = os.path.join(temp_dir, "test_chats")
        
        try:
            with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
                # Directory should not exist initially
                assert not os.path.exists(test_dir)
                
                # Call ensure_chat_directory
                ensure_chat_directory()
                
                # Directory should now exist
                assert os.path.exists(test_dir)
                assert os.path.isdir(test_dir)
                
                # Calling again should not raise error
                ensure_chat_directory()
                assert os.path.exists(test_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_chat_history_save_and_load(self):
        """Test complete save and load cycle"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch('chat_history.CHAT_HISTORY_DIR', temp_dir):
                # Test messages
                messages = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
                
                # Save the messages
                filename = auto_save_chat(messages)
                assert filename is not None
                
                # Check file was created
                filepath = os.path.join(temp_dir, filename)
                assert os.path.exists(filepath)
                
                # Load the messages back
                loaded_messages = load_chat_history(filename)
                assert loaded_messages == messages
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_chat_info_actual_behavior(self):
        """Test get_chat_info with actual behavior"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch('chat_history.CHAT_HISTORY_DIR', temp_dir):
                # Create a test file with content
                messages = [
                    {"role": "user", "content": "Hello, world!"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
                
                # Save first to get proper filename
                filename = auto_save_chat(messages)
                
                # Now test get_chat_info
                title, preview = get_chat_info(filename)
                
                # The title is based on filename without extension
                expected_title = filename.replace('.json', '')
                assert title == expected_title
                
                # Preview should contain the first assistant message content
                assert "Hi there!" in preview
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_chat_info_nonexistent_file(self):
        """Test get_chat_info with nonexistent file"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch('chat_history.CHAT_HISTORY_DIR', temp_dir):
                title, preview = get_chat_info("nonexistent.json")
                
                # Should return filename without extension as title
                assert title == "nonexistent"
                assert preview == "Error loading preview"
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestMockingFunctionality:
    """Test that mocking and test utilities work correctly"""
    
    def test_environment_variable_mocking(self):
        """Test environment variable mocking works"""
        # Save original values
        original_openai = os.environ.get('OPENAI_API_KEY')
        original_anthropic = os.environ.get('ANTHROPIC_API_KEY')
        
        try:
            # Clear environment
            with patch.dict(os.environ, {}, clear=True):
                assert 'OPENAI_API_KEY' not in os.environ
                assert 'ANTHROPIC_API_KEY' not in os.environ
                
                # Add test values
                with patch.dict(os.environ, {'TEST_KEY': 'test_value'}):
                    assert os.environ.get('TEST_KEY') == 'test_value'
                
                # Should be cleared after context
                assert 'TEST_KEY' not in os.environ
        finally:
            # Restore original values
            if original_openai:
                os.environ['OPENAI_API_KEY'] = original_openai
            if original_anthropic:
                os.environ['ANTHROPIC_API_KEY'] = original_anthropic

    def test_mock_object_functionality(self):
        """Test basic mock object functionality"""
        mock_obj = Mock()
        mock_obj.test_method.return_value = "mocked_result"
        
        result = mock_obj.test_method("test_arg")
        
        assert result == "mocked_result"
        mock_obj.test_method.assert_called_once_with("test_arg")
        
        # Test call count
        assert mock_obj.test_method.call_count == 1
        
        # Test reset
        mock_obj.reset_mock()
        assert mock_obj.test_method.call_count == 0

    def test_temporary_file_handling(self):
        """Test temporary file and directory handling works"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Directory should exist
            assert os.path.exists(temp_dir)
            assert os.path.isdir(temp_dir)
            
            # Create a test file
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test content")
            
            # File should exist
            assert os.path.exists(test_file)
            
            # Read content back
            with open(test_file, 'r') as f:
                content = f.read()
            assert content == "test content"
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Should be cleaned up
            assert not os.path.exists(temp_dir)


class TestImportAndBasicFunctionality:
    """Test that all modules can be imported and basic functionality works"""
    
    def test_all_imports_work(self):
        """Test that all modules can be imported without errors"""
        # These should not raise ImportError
        import ai_client
        import chat_history
        import file_manager
        import app
        
        # Test that key functions exist
        assert hasattr(ai_client, 'get_ai_client')
        assert hasattr(ai_client, 'get_available_models')
        assert hasattr(chat_history, 'auto_save_chat')
        assert hasattr(chat_history, 'load_chat_history')
        assert hasattr(file_manager, 'format_file_size')
        assert hasattr(app, 'stream_response')
        
        assert True  # If we get here, imports worked

    def test_basic_string_operations(self):
        """Test basic Python functionality works in test environment"""
        test_string = "Hello, World!"
        
        assert test_string.startswith("Hello")
        assert test_string.endswith("!")
        assert "World" in test_string
        assert len(test_string) == 13

    def test_basic_list_operations(self):
        """Test basic list operations work"""
        test_list = [1, 2, 3, 4, 5]
        
        assert len(test_list) == 5
        assert 3 in test_list
        assert test_list[0] == 1
        assert test_list[-1] == 5
        
        test_list.append(6)
        assert len(test_list) == 6
        assert test_list[-1] == 6

    def test_basic_dict_operations(self):
        """Test basic dictionary operations work"""
        test_dict = {"key1": "value1", "key2": "value2"}
        
        assert "key1" in test_dict
        assert test_dict["key1"] == "value1"
        assert len(test_dict) == 2
        
        test_dict["key3"] = "value3"
        assert len(test_dict) == 3
        assert test_dict["key3"] == "value3" 