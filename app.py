import streamlit as st
import os
from typing import Generator, Dict, Any

from ai_client import get_ai_client, get_available_models
from chat_history import (
    auto_save_chat, 
    load_chat_history, 
    get_saved_chats, 
    delete_chat_history, 
    get_chat_info
)

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

def create_file_reference(file_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create a file reference content block based on file type"""
    mime_type = file_info["mime_type"]
    
    if mime_type == "application/pdf" or mime_type == "text/plain":
        # Document block for supported file types
        return {
            "type": "document",
            "source": {
                "type": "file",
                "file_id": file_info["id"]
            },
            "filename": file_info["filename"]  # Keep for UI display
        }
    elif mime_type.startswith("image/"):
        # Image block for images
        return {
            "type": "image",
            "source": {
                "type": "file", 
                "file_id": file_info["id"]
            },
            "filename": file_info["filename"]  # Keep for UI display
        }
    else:
        # Container upload for other file types (CSV, JSON, etc.)
        return {
            "type": "container_upload",
            "source": {
                "type": "file",
                "file_id": file_info["id"]
            },
            "filename": file_info["filename"]  # Keep for UI display
        }

def handle_file_upload(uploaded_file, model) -> bool:
    """Handle file upload and add to session state. Returns True if successful."""
    import tempfile
    import os
    
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
    if "pending_files" not in st.session_state:
        st.session_state.pending_files = []
    
    file_ref = create_file_reference(file_info)
    st.session_state.pending_files.append(file_ref)

def format_file_size(size_bytes: int) -> str:
    """Format file size in a readable format."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f}KB"
    else:
        return f"{size_bytes/(1024*1024):.1f}MB"

# Configure Streamlit page
st.set_page_config(
    page_title="Simple Chat",
    page_icon="💬",
    layout="wide"
)



def stream_response(messages, model, thinking_enabled=False, thinking_budget=4000):
    """Stream response from AI provider using polymorphic client"""
    client = get_ai_client(model)
    
    try:
        for chunk in client.create_stream(messages, model, thinking_enabled=thinking_enabled, thinking_budget=thinking_budget):
            yield chunk
    except Exception as e:
        yield f"Error: {str(e)}"

def main():
    st.title("💬 Simple Chat")
    st.caption("🟢 OpenAI GPT • 🟣 Anthropic Claude • Auto-save conversations")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_chat_name" not in st.session_state:
        st.session_state.current_chat_name = None
    if "thinking_enabled" not in st.session_state:
        st.session_state.thinking_enabled = False
    if "thinking_budget" not in st.session_state:
        st.session_state.thinking_budget = 4000
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Sidebar for settings and chat management
    with st.sidebar:
        st.header("Settings")
        
        # Model selection with provider grouping
        st.subheader("🤖 AI Model")
        
        # Get available models from all providers
        try:
            available_models = get_available_models()
            
            # Create model options with provider labels
            model_options = []
            model_labels = []
            
            for provider, models in available_models.items():
                if models:  # Only show providers with available models
                    provider_name = provider.title()
                    for model_name in models:
                        model_options.append(model_name)
                        # Add provider prefix for display
                        if provider == "openai":
                            model_labels.append(f"🟢 {model_name}")
                        elif provider == "claude":
                            model_labels.append(f"🟣 {model_name}")
                        else:
                            model_labels.append(f"⚪ {model_name}")
            
            if model_options:
                # Default to gpt-4o if available, otherwise first available model
                default_index = 0
                if "gpt-4o" in model_options:
                    default_index = model_options.index("gpt-4o")
                
                selected_index = st.selectbox(
                    "Choose Model:",
                    range(len(model_options)),
                    index=default_index,
                    format_func=lambda x: model_labels[x],
                    help="🟢 OpenAI • 🟣 Claude"
                )
                
                model = model_options[selected_index]
            else:
                st.error("No AI models available. Check your API keys in .env file.")
                with st.expander("📋 Setup Instructions", expanded=True):
                    st.markdown("""
                    **To use this app, create a `.env` file with your API keys:**
                    
                    ```
                    # For OpenAI models (GPT-4o, GPT-4, etc.)
                    OPENAI_API_KEY=your_openai_api_key_here
                    
                    # For Claude models (Claude-3.5-Sonnet, etc.)
                    # Use either variable name:
                    ANTHROPIC_API_KEY=your_anthropic_api_key_here
                    # OR
                    CLAUDE_API_KEY=your_claude_api_key_here
                    ```
                    
                    **Get API keys from:**
                    - OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
                    - Anthropic: [console.anthropic.com](https://console.anthropic.com/)
                    
                    You need at least one API key to use the app.
                    """)
                model = "gpt-3.5-turbo"  # Fallback
                
        except Exception as e:
            st.warning(f"Error loading models: {e}")
            # Fallback to basic OpenAI models
            model = st.selectbox(
                "Choose Model (Fallback):",
                ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
                index=0
            )
        
        # Extended Thinking Controls (only for Claude 4 models)
        claude_thinking_models = ["claude-sonnet-4-20250514", "claude-opus-4-20250514"]
        if model in claude_thinking_models:
            st.divider()
            st.subheader("🧠 Extended Thinking")
            
            thinking_enabled = st.checkbox(
                "Enable Extended Thinking",
                value=st.session_state.thinking_enabled,
                help="Allow Claude to show its step-by-step reasoning process for complex problems"
            )
            
            if thinking_enabled != st.session_state.thinking_enabled:
                st.session_state.thinking_enabled = thinking_enabled
            
            if thinking_enabled:
                thinking_budget = st.slider(
                    "Thinking Budget (tokens)",
                    min_value=1024,
                    max_value=32000,
                    value=st.session_state.thinking_budget,
                    step=1024,
                    help="How many tokens Claude can use for thinking. Higher values allow deeper reasoning but cost more."
                )
                
                if thinking_budget != st.session_state.thinking_budget:
                    st.session_state.thinking_budget = thinking_budget
                
                st.caption(f"💰 Current budget: {thinking_budget:,} tokens")
        
        st.divider()
        
        # File Upload and Management (only for Claude models)
        if model.startswith("claude-"):
            st.subheader("📁 File Management")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload File",
                type=SUPPORTED_EXTENSIONS,
                help="Supports documents, code files, images, and data files"
            )
            
            # Handle file upload
            if uploaded_file is not None:
                # Use file name and size as simple duplicate check
                file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                
                # Initialize upload tracking
                if "last_uploaded_file" not in st.session_state:
                    st.session_state.last_uploaded_file = None
                
                # Only process if this is a different file than last time
                if st.session_state.last_uploaded_file != file_key:
                    st.session_state.last_uploaded_file = file_key
                    
                    if handle_file_upload(uploaded_file, model):
                        st.success(f"✅ Uploaded: {uploaded_file.name}")
                    # No st.rerun() - let Streamlit handle UI updates naturally
            
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

        st.divider()
        
        # Chat History Management
        st.header("💬 Chat History")
        
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True):
            # Auto-save current chat if it has messages
            if st.session_state.messages:
                filename = auto_save_chat(st.session_state.messages, st.session_state.current_chat_name)
                if filename and not st.session_state.current_chat_name:
                    st.session_state.current_chat_name = filename
            
            # Start new chat
            st.session_state.messages = []
            st.session_state.current_chat_name = None
            st.rerun()
        
        # Show current chat info
        if st.session_state.current_chat_name:
            current_title, _ = get_chat_info(st.session_state.current_chat_name)
            st.info(f"📝 {current_title}")
        elif st.session_state.messages:
            st.info("📝 New chat (auto-saves)")
        
        st.divider()
        
        # Load saved chats
        saved_chats = get_saved_chats()
        if saved_chats:
            st.subheader("💬 Recent Chats")
            
            for chat_file in saved_chats:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Get chat title
                    title, _ = get_chat_info(chat_file)
                    
                    # Highlight current chat
                    button_type = "primary" if chat_file == st.session_state.current_chat_name else "secondary"
                    
                    if st.button(title, key=f"load_{chat_file}", type=button_type, use_container_width=True):
                        # Auto-save current chat before switching
                        if st.session_state.messages and st.session_state.current_chat_name != chat_file:
                            auto_save_chat(st.session_state.messages, st.session_state.current_chat_name)
                        
                        # Load selected chat
                        loaded_messages = load_chat_history(chat_file)
                        if loaded_messages:
                            st.session_state.messages = loaded_messages
                            st.session_state.current_chat_name = chat_file
                            st.rerun()
                
                with col2:
                    # Delete button
                    if st.button("🗑️", key=f"delete_{chat_file}", help="Delete this chat"):
                        if delete_chat_history(chat_file):
                            if st.session_state.current_chat_name == chat_file:
                                st.session_state.current_chat_name = None
                                st.session_state.messages = []
                            st.rerun()
        else:
            st.info("💬 Start chatting to create your first conversation!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            if isinstance(content, str):
                st.markdown(content)
            elif isinstance(content, list):
                # Handle content blocks (text + files)
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            st.markdown(block.get("text", ""))
                        elif block.get("type") in ["document", "image", "container_upload"]:
                            # Show file attachment info
                            file_id = block.get("source", {}).get("file_id", "Unknown")
                            filename = block.get("filename", f"File {file_id}")
                            st.info(f"📎 {filename}")
                    else:
                        st.markdown(str(block))
            else:
                st.markdown(str(content))
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Prepare user message with files if any
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
        message_content = prompt if len(user_content) == 1 else user_content
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": message_content})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            # Show attached files
            if isinstance(message_content, list) and len(message_content) > 1:
                file_count = len(message_content) - 1
                st.caption(f"📎 {file_count} file{'s' if file_count != 1 else ''} attached")
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            update_buffer = ""
            chunk_counter = 0
            
            # Stream the response with smart buffering
            for chunk in stream_response(
                st.session_state.messages, 
                model,
                thinking_enabled=st.session_state.thinking_enabled,
                thinking_budget=st.session_state.thinking_budget
            ):
                full_response += chunk
                update_buffer += chunk
                chunk_counter += 1
                
                # Update display every 3 chunks OR when buffer gets large (smoother for code)
                if chunk_counter % 3 == 0 or len(update_buffer) > 30:
                    response_placeholder.markdown(full_response + "▌")
                    update_buffer = ""
            
            # Remove cursor and show final response
            response_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Auto-save the conversation after each exchange
        filename = auto_save_chat(st.session_state.messages, st.session_state.current_chat_name)
        if filename and not st.session_state.current_chat_name:
            st.session_state.current_chat_name = filename

if __name__ == "__main__":
    main()