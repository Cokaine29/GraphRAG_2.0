import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import GEMINI_API_KEY
import google.generativeai as genai

def main():
    print("Testing Gemini API...")
    try:
        # Force REST transport to avoid gRPC hang on Windows
        genai.configure(api_key=GEMINI_API_KEY, transport='rest')
        
        # Use a model with a higher daily limit
        model = genai.GenerativeModel("models/gemini-2.0-flash-lite")
        
        print(f"Model: models/gemini-2.0-flash-lite")
        
        response = model.generate_content("Hello Gemini! Please reply with a short greeting and confirm you are working.")
        
        print("\n--- Gemini Response ---")
        print(response.text)
        print("-----------------------")
        print("\nSuccess! The Gemini API is working properly.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
