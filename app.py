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

def create_file_reference(file_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create a file reference content block based on file type"""
    mime_type = file_info["mime_type"]
    filename = file_info["filename"].lower()
    
    # Code and text file extensions (now uploaded as text/plain)
    code_extensions = ('.py', '.pyw', '.js', '.jsx', '.ts', '.tsx', '.html', '.htm', '.css', '.scss', '.sass', '.md', '.txt', '.json', '.xml', '.yaml', '.yml', '.sql', '.sh', '.bat', '.ps1', '.php', '.rb', '.go', '.rs', '.cpp', '.c', '.h', '.hpp', '.java', '.kt', '.swift', '.r', '.m', '.pl', '.lua', '.vim', '.dockerfile')
    
    if (mime_type == "application/pdf" or 
        mime_type == "text/plain" or 
        filename.endswith(code_extensions)):
        # Document block for PDFs, text files, and code files (uploaded as text/plain)
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
                type=['pdf', 'txt', 'py', 'pyw', 'js', 'jsx', 'ts', 'tsx', 'html', 'htm', 'css', 'scss', 'sass', 'json', 'xml', 'yaml', 'yml', 'md', 'sql', 'sh', 'bat', 'php', 'rb', 'go', 'rs', 'cpp', 'c', 'h', 'hpp', 'java', 'kt', 'swift', 'r', 'm', 'pl', 'lua', 'vim', 'dockerfile', 'png', 'jpg', 'jpeg', 'gif', 'webp', 'csv', 'docx', 'xlsx'],
                help="Upload files to use with Claude. Supports PDFs, images, code files (Python, JavaScript, TypeScript, HTML, CSS, etc.), text files, and data files."
            )
            
            if uploaded_file is not None:
                try:
                    # Save uploaded file temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    # Upload to Claude Files API
                    client = get_ai_client(model)
                    if hasattr(client, 'upload_file'):
                        file_info = client.upload_file(temp_path)
                        
                        # Add to session state if not already there
                        if not any(f["id"] == file_info["id"] for f in st.session_state.uploaded_files):
                            st.session_state.uploaded_files.append(file_info)
                            st.success(f"✅ Uploaded: {file_info['filename']}")
                        
                        # Clean up temp file
                        import os
                        os.unlink(temp_path)
                    else:
                        st.error("File uploads not supported for this model")
                        
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
            
            # Display uploaded files
            if st.session_state.uploaded_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader("📄 Uploaded Files")
                with col2:
                    if st.button("🗑️ Clear All", help="Delete all uploaded files", type="secondary"):
                        try:
                            client = get_ai_client(model)
                            if hasattr(client, 'delete_file'):
                                deleted_count = 0
                                for file_info in st.session_state.uploaded_files:
                                    try:
                                        client.delete_file(file_info["id"])
                                        deleted_count += 1
                                    except:
                                        pass  # Continue deleting others
                                st.session_state.uploaded_files = []
                                st.session_state.pending_files = []
                                st.success(f"🗑️ Deleted {deleted_count} files")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to clear files: {str(e)}")
                
                files_to_remove = []
                for i, file_info in enumerate(st.session_state.uploaded_files):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        # File info with size
                        size_mb = file_info["size_bytes"] / (1024 * 1024)
                        st.text(f"{file_info['filename']} ({size_mb:.1f}MB)")
                        st.caption(f"📎 {file_info['mime_type']}")
                    
                    with col2:
                        # Add to message button
                        if st.button("➕", key=f"add_{i}", help="Add to next message"):
                            # Create file reference based on MIME type
                            file_ref = create_file_reference(file_info)
                            if "pending_files" not in st.session_state:
                                st.session_state.pending_files = []
                            st.session_state.pending_files.append(file_ref)
                            st.success(f"📎 Added {file_info['filename']} to next message")
                    
                    with col3:
                        # Delete button
                        if st.button("🗑️", key=f"delete_{i}", help="Delete file"):
                            try:
                                client = get_ai_client(model)
                                if hasattr(client, 'delete_file'):
                                    client.delete_file(file_info["id"])
                                files_to_remove.append(i)
                                st.success(f"🗑️ Deleted {file_info['filename']}")
                            except Exception as e:
                                st.error(f"Failed to delete: {str(e)}")
                
                # Remove deleted files from session state
                for i in reversed(files_to_remove):
                    st.session_state.uploaded_files.pop(i)
                
                if files_to_remove:
                    st.rerun()
                
                # Show help message
                st.info("💡 **Tip:** Click the ➕ button next to a file to attach it to your next message!")
            
            # Show pending file attachments
            if "pending_files" in st.session_state and st.session_state.pending_files:
                st.subheader("📎 Files for Next Message")
                for file_ref in st.session_state.pending_files:
                    if "filename" in file_ref:
                        st.info(f"📎 {file_ref['filename']}")

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
    
    # Show attachment status
    if st.session_state.uploaded_files and ("pending_files" not in st.session_state or not st.session_state.pending_files):
        st.warning("⚠️ You have uploaded files but none are attached to your next message. Click ➕ next to files in the sidebar to attach them!")
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Prepare user message with files if any
        user_content = []
        
        # Add text content
        user_content.append({
            "type": "text",
            "text": prompt
        })
        
        # Add pending files if any
        if "pending_files" in st.session_state and st.session_state.pending_files:
            st.info(f"📎 Attaching {len(st.session_state.pending_files)} file(s) to this message...")
            for file_ref in st.session_state.pending_files:
                # Remove filename key as it's not part of the API format
                api_file_ref = {k: v for k, v in file_ref.items() if k != "filename"}
                user_content.append(api_file_ref)
            
            # Clear pending files after adding them
            st.session_state.pending_files = []
        
        # Use simple text format if no files, otherwise use content blocks
        message_content = prompt if len(user_content) == 1 else user_content
        
        # Debug: Show message structure
        if len(user_content) > 1:
            st.info(f"🔍 Debug: Sending message with {len(user_content)} content blocks ({len(user_content)-1} files + 1 text)")
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": message_content})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            # Show attached files
            if isinstance(message_content, list) and len(message_content) > 1:
                for content_block in message_content[1:]:  # Skip first text block
                    if content_block.get("type") in ["document", "image", "container_upload"]:
                        file_id = content_block.get("source", {}).get("file_id", "")
                        # Find filename from uploaded files
                        filename = "Unknown file"
                        for f in st.session_state.uploaded_files:
                            if f["id"] == file_id:
                                filename = f["filename"]
                                break
                        st.info(f"📎 Attached: {filename}")
        
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