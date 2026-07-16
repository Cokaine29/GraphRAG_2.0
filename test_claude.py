import os
import sys

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.llm_client import LLMClient

def main():
    print("Testing Claude API...")
    try:
        llm = LLMClient(purpose="extraction")
        print(f"Provider: {llm.provider_name}")
        print(f"Model: {llm.model}")
        
        response = llm.generate("Hello Claude! Please reply with a short greeting and confirm you are working.")
        
        print("\n--- Claude Response ---")
        print(response)
        print("-----------------------")
        print("\nSuccess! The Claude API is working properly.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
