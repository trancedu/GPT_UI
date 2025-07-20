import openai
import os
from dotenv import load_dotenv
from typing import Iterator, List, Dict, Any
import httpx
from abc import ABC, abstractmethod

# Load environment variables from .env file
# Check for .env file in current directory and parent directories
load_dotenv()  # Current directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))  # Parent directory

class AIClient(ABC):
    """Abstract base class for AI clients"""
    
    @abstractmethod
    def create_stream(self, messages: List[Dict[str, str]], model: str, thinking_enabled: bool = False, thinking_budget: int = 4000) -> Iterator[str]:
        """Create a streaming response"""
        pass
    
    @abstractmethod
    def create_response(self, messages: List[Dict[str, str]], model: str, thinking_enabled: bool = False, thinking_budget: int = 4000) -> str:
        """Create a complete response (non-streaming)"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider"""
        pass

class OpenAIClient(AIClient):
    """OpenAI client implementation"""
    
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Try multiple initialization approaches to handle version compatibility
        try:
            # First try: Standard initialization
            self.client = openai.OpenAI(api_key=api_key)
        except TypeError as e:
            if "proxies" in str(e):
                try:
                    # Second try: Explicit httpx client without problematic arguments
                    http_client = httpx.Client()
                    self.client = openai.OpenAI(
                        api_key=api_key,
                        http_client=http_client
                    )
                except Exception:
                    try:
                        # Third try: Create client with minimal httpx configuration
                        http_client = httpx.Client(
                            timeout=httpx.Timeout(30.0),
                            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
                        )
                        self.client = openai.OpenAI(
                            api_key=api_key,
                            http_client=http_client
                        )
                    except Exception:
                        # Final fallback: Basic initialization (older openai versions)
                        self.client = openai.OpenAI(
                            api_key=api_key,
                            http_client=None
                        )
            else:
                # Re-raise if it's not a proxies-related error
                raise e
    
    def create_stream(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", thinking_enabled: bool = False, thinking_budget: int = 4000) -> Iterator[str]:
        """Create a streaming response from OpenAI"""
        # Note: OpenAI doesn't support extended thinking, so these parameters are ignored
        try:
            stream = self.client.chat.completions.create(
                messages=messages,
                model=model,
                stream=True,
            )
            
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
                    
        except Exception as e:
            yield f"Error: {e}"
    
    def create_response(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", thinking_enabled: bool = False, thinking_budget: int = 4000) -> str:
        """Create a complete response from OpenAI (non-streaming)"""
        # Note: OpenAI doesn't support extended thinking, so these parameters are ignored
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                stream=False,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error: {e}"
    
    def get_available_models(self) -> List[str]:
        """Get available OpenAI models"""
        return ["gpt-4o", "gpt-3.5-turbo"]

class ClaudeClient(AIClient):
    """Claude (Anthropic) client implementation"""
    
    def __init__(self):
        # Check for both possible API key environment variables
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable is not set")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package is required for Claude support. Install with: pip install anthropic")
    
    def create_stream(self, messages: List[Dict[str, str]], model: str = "claude-3-haiku-20240307", thinking_enabled: bool = False, thinking_budget: int = 4000) -> Iterator[str]:
        """Create a streaming response from Claude with optional thinking support"""
        try:
            # Convert OpenAI format messages to Claude format
            claude_messages = self._convert_messages_to_claude_format(messages)
            
            # Prepare request parameters
            stream_params = {
                "model": model,
                "max_tokens": 16000,  # Increased for thinking
                "messages": claude_messages,
                "stream": True
            }
            
            # Add thinking parameters if enabled
            if thinking_enabled:
                stream_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }
            
            # Stream the response
            response = self.client.messages.create(**stream_params)
            
            thinking_content = ""
            text_content = ""
            current_content_type = None
            
            for event in response:
                if event.type == "message_start":
                    continue
                elif event.type == "content_block_start":
                    current_content_type = event.content_block.type
                    if current_content_type == "thinking":
                        thinking_content = ""
                        # Yield thinking header
                        yield "\n🧠 **Claude is thinking:**\n\n"
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, 'thinking') and event.delta.thinking:
                        # Thinking content
                        thinking_chunk = event.delta.thinking
                        thinking_content += thinking_chunk
                        # Yield thinking in a code block for formatting
                        if thinking_chunk:
                            yield thinking_chunk
                    elif hasattr(event.delta, 'text') and event.delta.text:
                        # Regular text content
                        text_chunk = event.delta.text
                        text_content += text_chunk
                        # Add separator before text if we had thinking
                        if thinking_content and not text_content.startswith('\n'):
                            yield "\n\n💬 **Claude's response:**\n\n"
                            text_content = '\n\n💬 **Claude\'s response:**\n\n' + text_chunk
                        yield text_chunk
                elif event.type == "content_block_stop":
                    if current_content_type == "thinking":
                        # End of thinking block
                        yield "\n\n---\n"
                elif event.type == "message_stop":
                    break
                    
        except Exception as e:
            yield f"Error: {e}"
    
    def create_response(self, messages: List[Dict[str, str]], model: str = "claude-3-haiku-20240307", thinking_enabled: bool = False, thinking_budget: int = 4000) -> str:
        """Create a complete response from Claude (non-streaming) with optional thinking support"""
        try:
            # Convert OpenAI format messages to Claude format
            claude_messages = self._convert_messages_to_claude_format(messages)
            
            # Prepare request parameters
            params = {
                "model": model,
                "max_tokens": 16000,  # Increased for thinking
                "messages": claude_messages
            }
            
            # Add thinking parameters if enabled
            if thinking_enabled:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }
            
            response = self.client.messages.create(**params)
            
            # Combine thinking and text content
            result = ""
            
            for content_block in response.content:
                if content_block.type == "thinking":
                    result += f"\n🧠 **Claude is thinking:**\n\n{content_block.thinking}\n\n---\n\n💬 **Claude's response:**\n\n"
                elif content_block.type == "text":
                    result += content_block.text
            
            return result.strip()
            
        except Exception as e:
            return f"Error: {e}"
    
    def get_available_models(self) -> List[str]:
        """Get available Claude models"""
        return ["claude-sonnet-4-20250514"]
    
    def _convert_messages_to_claude_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert OpenAI format messages to Claude format"""
        claude_messages = []
        
        for msg in messages:
            # Claude uses 'user' and 'assistant' roles (same as OpenAI)
            role = msg["role"]
            if role == "system":
                # Claude doesn't have system messages in the same way
                # We'll prepend system content to the first user message
                continue
            
            claude_messages.append({
                "role": role,
                "content": msg["content"]
            })
        
        # Handle system messages by prepending to first user message
        system_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n\n"
        
        if system_content and claude_messages and claude_messages[0]["role"] == "user":
            claude_messages[0]["content"] = system_content + claude_messages[0]["content"]
        
        return claude_messages

# Provider registry
PROVIDERS = {
    "openai": OpenAIClient,
    "claude": ClaudeClient
}

# Model to provider mapping
MODEL_PROVIDERS = {
    # OpenAI models
    "gpt-4o": "openai",
    "gpt-3.5-turbo": "openai",
    
    # Claude models
    "claude-sonnet-4-20250514": "claude"
}

# Client instances cache
client_instances = {}

def get_ai_client(model: str) -> AIClient:
    """Get appropriate AI client for the given model"""
    provider = MODEL_PROVIDERS.get(model)
    if not provider:
        # Default to OpenAI for unknown models
        provider = "openai"
    
    # Use cached instance if available
    if provider not in client_instances:
        try:
            client_class = PROVIDERS[provider]
            client_instances[provider] = client_class()
        except Exception as e:
            # Fallback to OpenAI if provider fails
            if provider != "openai":
                print(f"Warning: Failed to initialize {provider} client: {e}")
                print("Falling back to OpenAI client...")
                client_instances[provider] = OpenAIClient()
            else:
                raise e
    
    return client_instances[provider]

def get_available_models() -> Dict[str, List[str]]:
    """Get all available models grouped by provider"""
    models = {}
    for provider_name, client_class in PROVIDERS.items():
        try:
            # Try to get models from each provider
            if provider_name not in client_instances:
                client_instances[provider_name] = client_class()
            models[provider_name] = client_instances[provider_name].get_available_models()
        except Exception as e:
            print(f"Warning: Could not load {provider_name} models: {e}")
            models[provider_name] = []
    
    return models

# Legacy function for backward compatibility
def get_openai_client():
    """Get OpenAI client instance (legacy compatibility)"""
    return get_ai_client("gpt-3.5-turbo") 