"""
llm_client.py - universal LLM interface for Gemini, Groq, Claude, and Ollama.

Usage:
    from src.llm_client import LLMClient
    llm = LLMClient()                        # uses config default
    llm = LLMClient(provider="claude")       # override provider
    response = llm.generate("your prompt")
    embedding = llm.embed("your text")
"""

import time
import os
import json
import logging
from typing import Optional, List
from src.config import cfg, GEMINI_API_KEY, GROQ_API_KEY, CLAUDE_API_KEY, OLLAMA_URL

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, purpose: str = "generation"):
        """
        purpose: "extraction" | "generation" - picks defaults from config
        provider: "gemini" | "groq" | "claude" | "ollama" - overrides config
        """
        if purpose == "extraction":
            default_provider = cfg.get("llm", {}).get("extraction_provider", "ollama")
            default_model    = cfg.get("llm", {}).get("extraction_model", "llama3")
        else:
            default_provider = cfg.get("llm", {}).get("generation_provider", "ollama")
            default_model    = cfg.get("llm", {}).get("generation_model", "llama3")

        self.provider    = provider or default_provider
        self.model       = model or default_model
        self.temperature = cfg.get("llm", {}).get("temperature", 0.0)
        self.max_retries = cfg.get("llm", {}).get("max_retries", 3)
        self.retry_delay = cfg.get("llm", {}).get("retry_delay_seconds", 2)
        self._client     = None
        self._setup()

    @property
    def provider_name(self) -> str:
        return self.provider

    def _setup(self) -> None:
        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY, transport='rest')
            self._client = genai.GenerativeModel(self.model)
            self._genai  = genai

        elif self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=GROQ_API_KEY)

        elif self.provider == "claude":
            import anthropic
            import httpx
            self._client = anthropic.Anthropic(
                api_key=CLAUDE_API_KEY,
                http_client=httpx.Client(verify=False)
            )
            
        elif self.provider == "ollama":
            self._ollama_url = OLLAMA_URL

        else:
            raise ValueError(f"Unknown provider: {self.provider}. Choose gemini | groq | claude | ollama")

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Send prompt, return response string. Retries on failure."""
        temp = temperature if temperature is not None else self.temperature

        for attempt in range(self.max_retries):
            try:
                return self._generate(prompt, temp)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(f"[{self.provider}] Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"[{self.provider}] All {self.max_retries} attempts failed: {e}")
                    raise RuntimeError(f"[{self.provider}] All {self.max_retries} attempts failed: {e}")

    def _generate(self, prompt: str, temperature: float) -> str:
        if self.provider == "gemini":
            config = self._genai.types.GenerationConfig(temperature=temperature)
            response = self._client.generate_content(prompt, generation_config=config)
            return response.text.strip()

        elif self.provider == "groq":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "claude":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

        elif self.provider == "ollama":
            import requests
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=600,
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()

    def embed(self, text: str) -> List[float]:
        """Return embedding vector for text."""
        # By default, use config embedding provider if it is Gemini
        embed_provider = cfg.get("llm", {}).get("embedding_provider", "gemini")
        if embed_provider == "gemini" or self.provider == "gemini":
            if not hasattr(self, "_genai"):
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                self._genai = genai
            model = cfg.get("llm", {}).get("embedding_model", "models/text-embedding-004")
            result = self._genai.embed_content(model=model, content=text)
            return result["embedding"]
        elif embed_provider == "ollama" or self.provider == "ollama":
            import requests
            model = cfg.get("llm", {}).get("embedding_model", "nomic-embed-text:latest")
            url = getattr(self, "_ollama_url", OLLAMA_URL)
            resp = requests.post(f"{url}/api/embeddings", json={"model": model, "prompt": text})
            resp.raise_for_status()
            return resp.json()["embedding"]
        else:
            # Fallback: sentence-transformers (local, no API needed)
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, "_st_model"):
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            return self._st_model.encode(text).tolist()

    def embed_batch(self, texts: List[str], delay: float = 0.5) -> List[List[float]]:
        """Embed a list of texts with rate-limit delay between calls."""
        embeddings = []
        for i, text in enumerate(texts):
            embeddings.append(self.embed(text))
            if i < len(texts) - 1:
                time.sleep(delay)
        return embeddings

