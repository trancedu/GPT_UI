import pytest
import os
import json
import tempfile
from unittest.mock import patch, mock_open
from chat_history import (
    ensure_chat_directory,
    auto_save_chat, 
    load_chat_history,
    get_saved_chats,
    delete_chat_history,
    get_chat_info,
    save_chat_history
)

class TestChatHistory:
    """Test suite for chat history functionality"""

    def test_ensure_chat_directory(self, temp_chat_directory):
        """Test that chat directory is created if it doesn't exist"""
        test_dir = os.path.join(temp_chat_directory, "test_chats")
        
        # Mock the CHAT_HISTORY_DIR constant
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            assert not os.path.exists(test_dir)
            ensure_chat_directory()
            assert os.path.exists(test_dir)

    def test_ensure_chat_directory_already_exists(self, temp_chat_directory):
        """Test that ensure_chat_directory works when directory already exists"""
        test_dir = os.path.join(temp_chat_directory, "existing_chats")
        os.makedirs(test_dir)
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            ensure_chat_directory()  # Should not raise an exception
            assert os.path.exists(test_dir)

    def test_auto_save_chat_empty_messages(self):
        """Test auto_save_chat with empty messages returns None"""
        result = auto_save_chat([])
        assert result is None

    def test_auto_save_chat_basic(self, temp_chat_directory, sample_messages):
        """Test basic auto-save functionality"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            filename = auto_save_chat(sample_messages)
            
            assert filename is not None
            assert filename.startswith("Hello, how are you")
            assert filename.endswith(".json")
            
            # Check file was created
            filepath = os.path.join(test_dir, filename)
            assert os.path.exists(filepath)

    def test_auto_save_chat_with_files(self, temp_chat_directory, sample_messages_with_files):
        """Test auto-save with messages containing files"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            filename = auto_save_chat(sample_messages_with_files)
            
            assert filename is not None
            assert "Analyze this document" in filename or "Chat with files" in filename

    def test_auto_save_chat_updates_existing(self, temp_chat_directory, sample_messages):
        """Test that auto_save_chat updates existing file"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        existing_filename = "existing_chat.json"
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            # First save
            filename = auto_save_chat(sample_messages, existing_filename)
            assert filename == existing_filename
            
            # Check file exists
            filepath = os.path.join(test_dir, filename)
            assert os.path.exists(filepath)

    def test_load_chat_history_success(self, temp_chat_directory, sample_messages):
        """Test successfully loading chat history"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        os.makedirs(test_dir)
        
        # Create a test file
        test_filename = "test_chat.json"
        test_filepath = os.path.join(test_dir, test_filename)
        with open(test_filepath, 'w') as f:
            json.dump(sample_messages, f)
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            loaded_messages = load_chat_history(test_filename)
            assert loaded_messages == sample_messages

    def test_load_chat_history_file_not_found(self, temp_chat_directory):
        """Test loading non-existent chat history returns empty list"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            loaded_messages = load_chat_history("nonexistent.json")
            assert loaded_messages == []

    def test_load_chat_history_invalid_json(self, temp_chat_directory):
        """Test loading invalid JSON returns empty list"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        os.makedirs(test_dir)
        
        # Create a file with invalid JSON
        test_filename = "invalid.json"
        test_filepath = os.path.join(test_dir, test_filename)
        with open(test_filepath, 'w') as f:
            f.write("invalid json content")
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            loaded_messages = load_chat_history(test_filename)
            assert loaded_messages == []

    def test_get_saved_chats(self, temp_chat_directory):
        """Test getting list of saved chats"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        os.makedirs(test_dir)
        
        # Create test chat files
        test_files = ["chat1.json", "chat2.json", "not_a_chat.txt"]
        for filename in test_files:
            with open(os.path.join(test_dir, filename), 'w') as f:
                f.write("{}")
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            saved_chats = get_saved_chats()
            
            # Should only return .json files, sorted in reverse order
            assert "chat1.json" in saved_chats
            assert "chat2.json" in saved_chats
            assert "not_a_chat.txt" not in saved_chats
            
            # Check sorting (newest first)
            assert saved_chats == sorted([f for f in saved_chats if f.endswith('.json')], reverse=True)

    def test_get_saved_chats_empty_directory(self, temp_chat_directory):
        """Test get_saved_chats with empty directory"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            saved_chats = get_saved_chats()
            assert saved_chats == []

    def test_delete_chat_history_success(self, temp_chat_directory):
        """Test successfully deleting chat history"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        os.makedirs(test_dir)
        
        # Create a test file
        test_filename = "delete_me.json"
        test_filepath = os.path.join(test_dir, test_filename)
        with open(test_filepath, 'w') as f:
            f.write("{}")
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            result = delete_chat_history(test_filename)
            assert result is True
            assert not os.path.exists(test_filepath)

    def test_delete_chat_history_file_not_found(self, temp_chat_directory):
        """Test deleting non-existent file returns False"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            result = delete_chat_history("nonexistent.json")
            assert result is False

    def test_get_chat_info(self, temp_chat_directory, sample_messages):
        """Test getting chat info (title and preview)"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        os.makedirs(test_dir)
        
        # Create a test file
        test_filename = "test_info.json"
        test_filepath = os.path.join(test_dir, test_filename)
        with open(test_filepath, 'w') as f:
            json.dump(sample_messages, f)
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            title, preview = get_chat_info(test_filename)
            
            assert "Hello, how are you" in title
            assert "Hello, how are you" in preview

    def test_get_chat_info_file_not_found(self, temp_chat_directory):
        """Test get_chat_info with non-existent file"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            title, preview = get_chat_info("nonexistent.json")
            assert title == "nonexistent.json"
            assert preview == "No messages"

    def test_save_chat_history_with_filename(self, temp_chat_directory, sample_messages):
        """Test save_chat_history with specific filename"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            result = save_chat_history(sample_messages, "custom_name.json")
            
            assert result == "custom_name.json"
            filepath = os.path.join(test_dir, "custom_name.json")
            assert os.path.exists(filepath)

    def test_save_chat_history_auto_name(self, temp_chat_directory, sample_messages):
        """Test save_chat_history with automatic naming"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            result = save_chat_history(sample_messages)
            
            assert result is not None
            assert result.endswith(".json")
            assert "Hello, how are you" in result

    def test_invalid_filename_characters(self, temp_chat_directory):
        """Test that invalid filename characters are handled properly"""
        test_dir = os.path.join(temp_chat_directory, "chats")
        
        # Message with characters that aren't valid in filenames
        messages = [{"role": "user", "content": "What's the <best> way to do this?"}]
        
        with patch('chat_history.CHAT_HISTORY_DIR', test_dir):
            filename = auto_save_chat(messages)
            
            assert filename is not None
            assert "<" not in filename
            assert ">" not in filename
            assert "?" not in filename
            # Should contain cleaned version
            assert "Whats the" in filename or "What" in filename 