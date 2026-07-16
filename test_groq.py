import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Set API key for testing
os.environ["GROQ_API_KEY"] = "gsk_YOUR_GROQ_API_KEY"
os.environ["LLM_PROVIDER"] = "groq"

# Mock config so it doesn't crash if env file is out of sync
import src.config
src.config.GROQ_API_KEY = "gsk_YOUR_GROQ_API_KEY"

from src.llm_client import LLMClient

def main():
    print("Testing Groq API...")
    try:
        # Create an instance directly using Groq as provider
        llm = LLMClient(purpose="extraction")
        
        # Override provider and model for the test just in case config is set to something else
        llm.provider = "groq"
        llm.model = "llama-3.3-70b-versatile"
        llm._setup()
        
        print(f"Provider: {llm.provider}")
        print(f"Model: {llm.model}")
        
        response = llm.generate("Hello Groq! Please reply with a short greeting and confirm you are working.")
        
        print("\n--- Groq Response ---")
        print(response)
        print("-----------------------")
        print("\nSuccess! The Groq API is working properly.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
