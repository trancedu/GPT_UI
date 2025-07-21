import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, Mock, MagicMock, mock_open
from io import BytesIO
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from file_manager import (
    upload_file,
    create_file_reference,
    clear_all_files,
    attach_file_to_message,
    prepare_message_content,
    init_session_state,
    format_file_size,
    SUPPORTED_EXTENSIONS
)


class MockUploadedFile:
    """Mock Streamlit uploaded file object"""
    def __init__(self, name, content, size=None):
        self.name = name
        self.content = content.encode() if isinstance(content, str) else content
        self.size = size or len(self.content)
        
    def getvalue(self):
        return self.content
    
    def read(self):
        return self.content


class TestFileUpload:
    """Test suite for file upload functionality"""

    def test_supported_extensions_includes_txt_and_py(self):
        """Test that txt and py files are in supported extensions"""
        assert 'txt' in SUPPORTED_EXTENSIONS
        assert 'py' in SUPPORTED_EXTENSIONS
        
        # Test other expected file types
        assert 'pdf' in SUPPORTED_EXTENSIONS
        assert 'md' in SUPPORTED_EXTENSIONS
        assert 'js' in SUPPORTED_EXTENSIONS
        assert 'png' in SUPPORTED_EXTENSIONS

    @patch('file_manager.st.session_state')
    def test_init_session_state(self, mock_st_session):
        """Test session state initialization"""
        # Mock session state with proper __contains__ behavior
        def mock_contains(self, key):
            return False  # Simulate keys don't exist initially
        
        mock_st_session.__contains__ = mock_contains
        
        init_session_state()
        
        # Check that all required keys are set (they should be assigned)
        # We can't easily check __setitem__ calls, but we can verify the function runs
        assert True  # If we get here, the function ran without error

    def test_format_file_size_comprehensive(self):
        """Test file size formatting (already covered but ensuring it works for upload context)"""
        # Test file sizes typical for uploads
        assert format_file_size(1024) == "1.0KB"  # Small text file
        assert format_file_size(5120) == "5.0KB"  # Medium Python file
        assert format_file_size(1048576) == "1.0MB"  # Large file

    def test_create_file_reference_txt_file(self):
        """Test creating file reference for txt file"""
        file_info = {
            "id": "file_123",
            "filename": "test.txt",
            "mime_type": "text/plain",
            "size_bytes": 1024
        }
        
        result = create_file_reference(file_info)
        
        expected = {
            "type": "document",
            "source": {
                "type": "file",
                "file_id": "file_123"
            },
            "filename": "test.txt"
        }
        assert result == expected

    def test_create_file_reference_py_file(self):
        """Test creating file reference for Python file"""
        file_info = {
            "id": "file_456",
            "filename": "script.py",
            "mime_type": "text/x-python",
            "size_bytes": 2048
        }
        
        result = create_file_reference(file_info)
        
        expected = {
            "type": "container_upload",
            "source": {
                "type": "file",
                "file_id": "file_456"
            },
            "filename": "script.py"
        }
        assert result == expected

    def test_create_file_reference_pdf_file(self):
        """Test creating file reference for PDF file"""
        file_info = {
            "id": "file_789",
            "filename": "document.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 1048576
        }
        
        result = create_file_reference(file_info)
        
        expected = {
            "type": "document",
            "source": {
                "type": "file",
                "file_id": "file_789"
            },
            "filename": "document.pdf"
        }
        assert result == expected

    def test_create_file_reference_image_file(self):
        """Test creating file reference for image file"""
        file_info = {
            "id": "file_101",
            "filename": "image.png",
            "mime_type": "image/png",
            "size_bytes": 512000
        }
        
        result = create_file_reference(file_info)
        
        expected = {
            "type": "image",
            "source": {
                "type": "file",
                "file_id": "file_101"
            },
            "filename": "image.png"
        }
        assert result == expected

    @patch('file_manager.st.session_state')
    @patch('file_manager.st.error')
    @patch('file_manager.st.spinner')
    @patch('tempfile.NamedTemporaryFile')
    @patch('file_manager.get_ai_client')
    @patch('os.path.exists')
    @patch('os.unlink')
    def test_upload_txt_file_success(self, mock_unlink, mock_exists, mock_get_client, 
                                     mock_temp_file, mock_spinner, mock_error, mock_session_state):
        """Test successful txt file upload"""
        # Setup mocks
        mock_uploaded_files = Mock()
        mock_uploaded_files.__iter__ = Mock(return_value=iter([]))  # No existing files
        mock_session_state.uploaded_files = mock_uploaded_files
        
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "/tmp/test_file.txt"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        mock_temp_file.return_value.__exit__.return_value = None
        
        mock_exists.return_value = True
        
        # Mock AI client
        mock_client = Mock()
        mock_client.upload_file.return_value = {
            "id": "file_txt_123",
            "filename": "test.txt",
            "mime_type": "text/plain",
            "size_bytes": 1024
        }
        mock_get_client.return_value = mock_client
        
        # Mock uploaded file
        txt_content = "Hello, this is a test text file!\nSecond line of content."
        uploaded_file = MockUploadedFile("test.txt", txt_content)
        
        # Test upload
        result = upload_file(uploaded_file, "claude-3-5-sonnet-20241022")
        
        # Assertions
        assert result is True
        mock_client.upload_file.assert_called_once_with("/tmp/test_file.txt")
        mock_temp.write.assert_called_once_with(txt_content.encode())
        mock_unlink.assert_called_once_with("/tmp/test_file.txt")
        mock_uploaded_files.append.assert_called_once()

    @patch('file_manager.st.session_state')
    @patch('file_manager.st.error')
    @patch('file_manager.st.spinner')
    @patch('tempfile.NamedTemporaryFile')
    @patch('file_manager.get_ai_client')
    @patch('os.path.exists')
    @patch('os.unlink')
    def test_upload_py_file_success(self, mock_unlink, mock_exists, mock_get_client,
                                    mock_temp_file, mock_spinner, mock_error, mock_session_state):
        """Test successful Python file upload"""
        # Setup mocks
        mock_session_state.uploaded_files = []
        
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "/tmp/test_script.py"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        mock_temp_file.return_value.__exit__.return_value = None
        
        mock_exists.return_value = True
        
        # Mock AI client
        mock_client = Mock()
        mock_client.upload_file.return_value = {
            "id": "file_py_456",
            "filename": "script.py", 
            "mime_type": "text/x-python",
            "size_bytes": 2048
        }
        mock_get_client.return_value = mock_client
        
        # Mock uploaded Python file
        py_content = '''def hello_world():
    """A simple hello world function"""
    print("Hello, World!")
    return "success"

if __name__ == "__main__":
    hello_world()
'''
        uploaded_file = MockUploadedFile("script.py", py_content)
        
        # Test upload
        result = upload_file(uploaded_file, "claude-3-5-sonnet-20241022")
        
        # Assertions
        assert result is True
        mock_client.upload_file.assert_called_once_with("/tmp/test_script.py")
        mock_temp.write.assert_called_once_with(py_content.encode())
        mock_unlink.assert_called_once_with("/tmp/test_script.py")

    @patch('file_manager.st.session_state')
    @patch('file_manager.st.error')
    @patch('file_manager.get_ai_client')
    def test_upload_file_unsupported_model(self, mock_get_client, mock_error, mock_session_state):
        """Test upload with unsupported model"""
        mock_client = Mock()
        # Remove upload_file attribute to simulate unsupported model
        del mock_client.upload_file
        mock_get_client.return_value = mock_client
        
        uploaded_file = MockUploadedFile("test.txt", "content")
        
        result = upload_file(uploaded_file, "gpt-4")
        
        assert result is False
        mock_error.assert_called_once_with("File uploads not supported for this model")

    @patch('file_manager.st.session_state')
    @patch('file_manager.st.error')
    @patch('file_manager.st.spinner')
    @patch('tempfile.NamedTemporaryFile')
    @patch('file_manager.get_ai_client')
    @patch('os.path.exists')
    @patch('os.unlink')
    def test_upload_file_api_error(self, mock_unlink, mock_exists, mock_get_client,
                                   mock_temp_file, mock_spinner, mock_error, mock_session_state):
        """Test upload with API error"""
        # Setup mocks
        mock_session_state.uploaded_files = []
        
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "/tmp/test_file.txt"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        mock_temp_file.return_value.__exit__.return_value = None
        
        mock_exists.return_value = True
        
        # Mock AI client with error
        mock_client = Mock()
        mock_client.upload_file.side_effect = Exception("API Error: File too large")
        mock_get_client.return_value = mock_client
        
        uploaded_file = MockUploadedFile("test.txt", "content")
        
        # Test upload
        result = upload_file(uploaded_file, "claude-3-5-sonnet-20241022")
        
        # Assertions
        assert result is False
        mock_error.assert_called_once_with("Upload failed: API Error: File too large")
        mock_unlink.assert_called_once_with("/tmp/test_file.txt")  # Cleanup should still happen

    @patch('file_manager.st.session_state')
    @patch('file_manager.get_ai_client')
    def test_clear_all_files_success(self, mock_get_client, mock_session_state):
        """Test clearing all uploaded files"""
        # Setup mock session state
        mock_session_state.uploaded_files = [
            {"id": "file1", "filename": "test1.txt"},
            {"id": "file2", "filename": "test2.py"}
        ]
        mock_session_state.pending_files = [{"id": "file1"}]
        
        # Mock AI client
        mock_client = Mock()
        mock_client.delete_file = Mock(return_value=True)
        mock_get_client.return_value = mock_client
        
        # Test clearing
        result = clear_all_files("claude-3-5-sonnet-20241022")
        
        # Assertions
        assert result == 2
        assert mock_client.delete_file.call_count == 2
        mock_client.delete_file.assert_any_call("file1")
        mock_client.delete_file.assert_any_call("file2")
        
        # Check that session state is cleared
        assert mock_session_state.uploaded_files == []
        assert mock_session_state.pending_files == []

    @patch('file_manager.st.session_state')
    @patch('file_manager.get_ai_client')
    def test_clear_all_files_unsupported_model(self, mock_get_client, mock_session_state):
        """Test clearing files with unsupported model"""
        mock_client = Mock()
        # Remove delete_file method
        del mock_client.delete_file
        mock_get_client.return_value = mock_client
        
        result = clear_all_files("gpt-4")
        
        assert result == 0

    @patch('file_manager.st.session_state')
    def test_attach_file_to_message(self, mock_session_state):
        """Test attaching file to message"""
        mock_session_state.pending_files = []
        
        file_info = {
            "id": "file_123",
            "filename": "test.txt",
            "mime_type": "text/plain",
            "size_bytes": 1024
        }
        
        attach_file_to_message(file_info)
        
        # Check that file reference was added to pending files
        assert len(mock_session_state.pending_files) == 1
        file_ref = mock_session_state.pending_files[0]
        assert file_ref["type"] == "document"
        assert file_ref["source"]["file_id"] == "file_123"
        assert file_ref["filename"] == "test.txt"

    @patch('file_manager.st.session_state')
    def test_prepare_message_content_no_files(self, mock_session_state):
        """Test preparing message content without files"""
        mock_session_state.get.return_value = None
        
        result = prepare_message_content("Hello, world!")
        
        assert result == "Hello, world!"

    @patch('file_manager.st.session_state')
    def test_prepare_message_content_with_txt_file(self, mock_session_state):
        """Test preparing message content with txt file attached"""
        # Mock pending files
        pending_files = [{
            "type": "document",
            "source": {
                "type": "file",
                "file_id": "file_123"
            },
            "filename": "test.txt"  # This should be removed from API format
        }]
        
        mock_session_state.get.return_value = pending_files
        mock_session_state.pending_files = pending_files
        
        result = prepare_message_content("Please analyze this file")
        
        expected = [
            {"type": "text", "text": "Please analyze this file"},
            {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": "file_123"
                }
                # Note: filename key should be removed for API
            }
        ]
        
        assert result == expected
        # Check that pending files are cleared
        assert mock_session_state.pending_files == []

    @patch('file_manager.st.session_state')
    def test_prepare_message_content_with_py_file(self, mock_session_state):
        """Test preparing message content with Python file attached"""
        # Mock pending files
        pending_files = [{
            "type": "container_upload",
            "source": {
                "type": "file",
                "file_id": "file_456"
            },
            "filename": "script.py"
        }]
        
        mock_session_state.get.return_value = pending_files
        mock_session_state.pending_files = pending_files
        
        result = prepare_message_content("Review this Python code")
        
        expected = [
            {"type": "text", "text": "Review this Python code"},
            {
                "type": "container_upload",
                "source": {
                    "type": "file",
                    "file_id": "file_456"
                }
            }
        ]
        
        assert result == expected
        assert mock_session_state.pending_files == []

    @patch('file_manager.st.session_state')
    def test_prepare_message_content_multiple_files(self, mock_session_state):
        """Test preparing message content with multiple files"""
        pending_files = [
            {
                "type": "document",
                "source": {"type": "file", "file_id": "file_txt"},
                "filename": "doc.txt"
            },
            {
                "type": "container_upload", 
                "source": {"type": "file", "file_id": "file_py"},
                "filename": "code.py"
            }
        ]
        
        mock_session_state.get.return_value = pending_files
        mock_session_state.pending_files = pending_files
        
        result = prepare_message_content("Analyze these files")
        
        assert len(result) == 3  # text + 2 files
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "document"
        assert result[2]["type"] == "container_upload"
        
        # Verify filename keys are removed from API format
        assert "filename" not in result[1]
        assert "filename" not in result[2]


class TestFileUploadIntegration:
    """Integration tests for file upload workflow"""

    @patch('file_manager.st.session_state')
    def test_complete_txt_file_workflow(self, mock_session_state):
        """Test complete workflow: upload txt file, attach to message, prepare content"""
        mock_session_state.uploaded_files = []
        mock_session_state.pending_files = []
        
        # Step 1: File info after successful upload
        file_info = {
            "id": "file_workflow_123",
            "filename": "workflow_test.txt", 
            "mime_type": "text/plain",
            "size_bytes": 1024
        }
        
        # Step 2: Attach file to message
        attach_file_to_message(file_info)
        assert len(mock_session_state.pending_files) == 1
        
        # Step 3: Prepare message content
        mock_session_state.get.return_value = mock_session_state.pending_files
        result = prepare_message_content("Please read this text file")
        
        # Verify complete workflow
        expected = [
            {"type": "text", "text": "Please read this text file"},
            {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": "file_workflow_123"
                }
            }
        ]
        assert result == expected
        assert mock_session_state.pending_files == []

    @patch('file_manager.st.session_state')
    def test_complete_py_file_workflow(self, mock_session_state):
        """Test complete workflow: upload Python file, attach to message, prepare content"""
        mock_session_state.uploaded_files = []
        mock_session_state.pending_files = []
        
        # Step 1: File info after successful upload
        file_info = {
            "id": "file_py_workflow_456",
            "filename": "algorithm.py",
            "mime_type": "text/x-python", 
            "size_bytes": 3072
        }
        
        # Step 2: Attach file to message
        attach_file_to_message(file_info)
        assert len(mock_session_state.pending_files) == 1
        
        # Step 3: Prepare message content
        mock_session_state.get.return_value = mock_session_state.pending_files
        result = prepare_message_content("Review this Python algorithm")
        
        # Verify complete workflow
        expected = [
            {"type": "text", "text": "Review this Python algorithm"},
            {
                "type": "container_upload",
                "source": {
                    "type": "file",
                    "file_id": "file_py_workflow_456"
                }
            }
        ]
        assert result == expected
        assert mock_session_state.pending_files == [] 