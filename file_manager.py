import streamlit as st
import tempfile
import os
from typing import Dict, Any, List
from ai_client import get_ai_client

# File upload constants
SUPPORTED_EXTENSIONS = [
    # Documents
    'pdf', 'txt', 'md',
    # Code files  
    'py', 'js', 'html', 'css', 'json', 'xml', 'yaml', 'sql',
    'java', 'cpp', 'c', 'h', 'go', 'rs', 'php', 'rb', 'kt', 'swift',
    # Images
    'png', 'jpg', 'jpeg', 'gif', 'webp',
    # Data
    'csv', 'xlsx', 'docx'
]

def init_session_state():
    """Initialize file-related session state."""
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "pending_files" not in st.session_state:
        st.session_state.pending_files = []
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None

def create_file_reference(file_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create a file reference content block based on file type."""
    mime_type = file_info["mime_type"]
    
    if mime_type == "application/pdf" or mime_type == "text/plain":
        return {
            "type": "document",
            "source": {
                "type": "file",
                "file_id": file_info["id"]
            },
            "filename": file_info["filename"]  # Keep for UI display
        }
    elif mime_type.startswith("image/"):
        return {
            "type": "image",
            "source": {
                "type": "file", 
                "file_id": file_info["id"]
            },
            "filename": file_info["filename"]  # Keep for UI display
        }
    else:
        return {
            "type": "container_upload",
            "source": {
                "type": "file",
                "file_id": file_info["id"]
            },
            "filename": file_info["filename"]  # Keep for UI display
        }

def format_file_size(size_bytes: int) -> str:
    """Format file size in a readable format."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f}KB"
    else:
        return f"{size_bytes/(1024*1024):.1f}MB"

def upload_file(uploaded_file, model) -> bool:
    """Handle file upload. Returns True if successful."""
    temp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Upload to Claude Files API
        client = get_ai_client(model)
        if not hasattr(client, 'upload_file'):
            st.error("File uploads not supported for this model")
            return False
        
        # Show upload progress
        with st.spinner(f"Uploading {uploaded_file.name}..."):
            file_info = client.upload_file(temp_path)
        
        # Add to session state if not already there
        if not any(f["id"] == file_info["id"] for f in st.session_state.uploaded_files):
            st.session_state.uploaded_files.append(file_info)
            return True
        
        return False
        
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return False
        
    finally:
        # Always clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def clear_all_files(model) -> int:
    """Clear all uploaded files. Returns number of files deleted."""
    try:
        client = get_ai_client(model)
        if not hasattr(client, 'delete_file'):
            return 0
            
        deleted_count = 0
        for file_info in st.session_state.uploaded_files:
            try:
                client.delete_file(file_info["id"])
                deleted_count += 1
            except:
                continue
                
        st.session_state.uploaded_files = []
        st.session_state.pending_files = []
        return deleted_count
        
    except Exception:
        return 0

def attach_file_to_message(file_info):
    """Attach a file to the next message."""
    file_ref = create_file_reference(file_info)
    st.session_state.pending_files.append(file_ref)

def render_file_manager(model):
    """Render the complete file management UI."""
    if not model.startswith("claude-"):
        return
    
    st.subheader("📁 File Management")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload File",
        type=SUPPORTED_EXTENSIONS,
        help="Supports documents, code files, images, and data files"
    )
    
    # Handle file upload with duplicate prevention
    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.last_uploaded_file != file_key:
            st.session_state.last_uploaded_file = file_key
            if upload_file(uploaded_file, model):
                st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📄 Uploaded Files")
        with col2:
            if st.button("🗑️ Clear All", help="Delete all uploaded files", type="secondary"):
                deleted_count = clear_all_files(model)
                if deleted_count > 0:
                    st.success(f"🗑️ Deleted {deleted_count} files")
        
        files_to_remove = []
        for i, file_info in enumerate(st.session_state.uploaded_files):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.text(f"{file_info['filename']} ({format_file_size(file_info['size_bytes'])})")
                st.caption(f"📎 {file_info['mime_type']}")
            
            with col2:
                if st.button("➕", key=f"add_{i}", help="Attach to next message"):
                    attach_file_to_message(file_info)
            
            with col3:
                if st.button("🗑️", key=f"delete_{i}", help="Delete file"):
                    try:
                        client = get_ai_client(model)
                        if hasattr(client, 'delete_file'):
                            client.delete_file(file_info["id"])
                            files_to_remove.append(i)
                    except Exception as e:
                        st.error(f"Delete failed: {str(e)}")
        
        # Remove deleted files from session state
        for i in reversed(files_to_remove):
            st.session_state.uploaded_files.pop(i)
    
    # Show pending file attachments
    if st.session_state.get("pending_files"):
        st.subheader("📎 Ready to Send")
        for file_ref in st.session_state.pending_files:
            if "filename" in file_ref:
                st.success(f"📎 {file_ref['filename']}")

def prepare_message_content(prompt: str):
    """Prepare message content with files if any."""
    user_content = [{"type": "text", "text": prompt}]
    
    # Add pending files if any
    if st.session_state.get("pending_files"):
        for file_ref in st.session_state.pending_files:
            # Remove filename key as it's not part of the API format
            api_file_ref = {k: v for k, v in file_ref.items() if k != "filename"}
            user_content.append(api_file_ref)
        
        # Clear pending files after adding them
        st.session_state.pending_files = []
    
    # Use simple text format if no files, otherwise use content blocks
    return prompt if len(user_content) == 1 else user_content 