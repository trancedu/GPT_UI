#!/usr/bin/env python3
"""
Example demonstrating Claude web search functionality.

This example shows how to use the web search tool with Claude models.
"""

import os
from ai_client import get_ai_client

def main():
    """Demonstrate web search functionality with Claude"""
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("CLAUDE_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable")
        print("   You can create a .env file with your API key:")
        print("   ANTHROPIC_API_KEY=your_api_key_here")
        return
    
    try:
        # Get Claude client
        client = get_ai_client("claude-sonnet-4-20250514")
        
        print("🔍 Claude Web Search Example")
        print("=" * 40)
        
        # Example 1: Basic web search
        print("\n📝 Example 1: Basic Web Search")
        print("-" * 30)
        
        web_search_tool = client.create_web_search_tool(max_uses=2)
        tools = [web_search_tool]
        
        messages = [
            {"role": "user", "content": "What are the latest developments in AI research this week?"}
        ]
        
        print("🤖 Claude with Web Search:")
        response = client.create_response(messages, "claude-sonnet-4-20250514", tools=tools)
        print(response)
        
        # Example 2: Web search with domain restrictions
        print("\n\n📝 Example 2: Web Search with Domain Restrictions")
        print("-" * 50)
        
        restricted_search_tool = client.create_web_search_tool(
            max_uses=1,
            allowed_domains=["news.ycombinator.com", "arxiv.org", "github.com"],
            user_location="San Francisco, CA"
        )
        tools = [restricted_search_tool]
        
        messages = [
            {"role": "user", "content": "Find recent machine learning papers or discussions"}
        ]
        
        print("🤖 Claude with Restricted Web Search:")
        response = client.create_response(messages, "claude-sonnet-4-20250514", tools=tools)
        print(response)
        
        # Example 3: Streaming with web search
        print("\n\n📝 Example 3: Streaming Web Search")
        print("-" * 35)
        
        streaming_tool = client.create_web_search_tool(max_uses=1)
        tools = [streaming_tool]
        
        messages = [
            {"role": "user", "content": "What's the current weather in New York City?"}
        ]
        
        print("🤖 Claude Streaming with Web Search:")
        for chunk in client.create_stream(messages, "claude-sonnet-4-20250514", tools=tools):
            print(chunk, end="", flush=True)
        print()  # New line at the end
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you have:")
        print("   1. Set your ANTHROPIC_API_KEY environment variable")
        print("   2. Installed the anthropic package: pip install anthropic")

if __name__ == "__main__":
    main() 