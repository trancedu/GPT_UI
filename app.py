import streamlit as st
import os
from typing import Generator

from ai_client import get_ai_client, get_available_models
from chat_history import (
    auto_save_chat, 
    load_chat_history, 
    get_saved_chats, 
    delete_chat_history, 
    get_chat_info
)
import file_manager

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
    
    # Initialize file-related session state
    file_manager.init_session_state()

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
        
        # File Management
        file_manager.render_file_manager(model)
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
        # Prepare message content with files if any
        message_content = file_manager.prepare_message_content(prompt)
        
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