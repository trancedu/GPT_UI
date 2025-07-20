import os
import json
import datetime
from typing import List, Dict, Tuple, Optional

# Chat history storage directory (relative to the main folder)
CHAT_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_chats")

def ensure_chat_directory():
    """Ensure the chat history directory exists"""
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)

def auto_save_chat(messages: List[Dict[str, str]], current_filename: Optional[str] = None) -> Optional[str]:
    """
    Automatically save chat history with intelligent naming
    
    Args:
        messages: List of conversation messages
        current_filename: Current filename if updating existing chat
        
    Returns:
        str: Filename of saved chat, or None if failed
    """
    if not messages:
        return None
    
    ensure_chat_directory()
    
    # Generate chat title from first user message (or use timestamp)
    chat_title = None
    for msg in messages:
        if msg["role"] == "user":
            # Use first 30 characters of first user message as title
            chat_title = msg["content"][:30].strip()
            # Replace invalid filename characters
            chat_title = "".join(c for c in chat_title if c.isalnum() or c in (' ', '-', '_')).strip()
            break
    
    if not chat_title:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_title = f"chat_{timestamp}"
    
    filename = current_filename or f"{chat_title}.json"
    
    # Ensure unique filename
    counter = 1
    original_filename = filename
    filepath = os.path.join(CHAT_HISTORY_DIR, filename)
    while os.path.exists(filepath) and not current_filename:
        name_without_ext = original_filename.replace('.json', '')
        filename = f"{name_without_ext}_{counter}.json"
        filepath = os.path.join(CHAT_HISTORY_DIR, filename)
        counter += 1
    
    try:
        chat_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "title": chat_title,
            "messages": messages
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        
        return filename
    except Exception as e:
        try:
            import streamlit as st
            st.error(f"Error auto-saving chat: {e}")
        except ImportError:
            print(f"Error auto-saving chat: {e}")
        return None

def load_chat_history(filename: str) -> List[Dict[str, str]]:
    """
    Load chat history from a JSON file
    
    Args:
        filename: Name of the chat file to load
        
    Returns:
        List of message dictionaries
    """
    filepath = os.path.join(CHAT_HISTORY_DIR, filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        return chat_data.get("messages", [])
    except Exception as e:
        try:
            import streamlit as st
            st.error(f"Error loading chat: {e}")
        except ImportError:
            print(f"Error loading chat: {e}")
        return []

def get_saved_chats() -> List[str]:
    """
    Get list of saved chat files, sorted by modification time (newest first)
    
    Returns:
        List of chat filenames
    """
    try:
        ensure_chat_directory()
        files = [f for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith('.json')]
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(CHAT_HISTORY_DIR, x)), reverse=True)
        return files
    except Exception:
        return []

def delete_chat_history(filename: str) -> bool:
    """
    Delete a saved chat file
    
    Args:
        filename: Name of the chat file to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    filepath = os.path.join(CHAT_HISTORY_DIR, filename)
    try:
        os.remove(filepath)
        return True
    except Exception as e:
        try:
            import streamlit as st
            st.error(f"Error deleting chat: {e}")
        except ImportError:
            print(f"Error deleting chat: {e}")
        return False

def get_chat_info(filename: str) -> Tuple[str, str]:
    """
    Get chat title and preview from a saved chat file
    
    Args:
        filename: Name of the chat file
        
    Returns:
        Tuple of (title, preview)
    """
    try:
        filepath = os.path.join(CHAT_HISTORY_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        # Get title from chat data or generate from first user message
        title = chat_data.get("title")
        if not title:
            messages = chat_data.get("messages", [])
            for msg in messages:
                if msg["role"] == "user":
                    title = msg["content"][:30].strip()
                    break
            if not title:
                title = filename.replace('.json', '')
        
        # Get preview (first user message)
        messages = chat_data.get("messages", [])
        preview = "Empty chat"
        for msg in messages:
            if msg["role"] == "user":
                preview = msg["content"][:50]
                if len(msg["content"]) > 50:
                    preview += "..."
                break
        
        return title, preview
    except Exception:
        return filename.replace('.json', ''), "Error loading preview"

# Legacy function for backward compatibility
def save_chat_history(messages: List[Dict[str, str]], filename: Optional[str] = None) -> Optional[str]:
    """Legacy function - now calls auto_save_chat"""
    return auto_save_chat(messages, filename) 