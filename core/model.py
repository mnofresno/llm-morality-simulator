"""Local LLM model interface for behavior experiments."""

import os
from typing import Optional
from llama_cpp import Llama


class LocalLLM:
    """Generic interface for local LLM models using llama-cpp-python."""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: Optional[int] = None):
        """
        Initialize local LLM model.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_threads: Number of threads (None for auto)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False
        )
    
    def infer(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stop: Optional[list] = None
    ) -> str:
        """
        Generate response from the model.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            
        Returns:
            Generated response text
        """
        if stop is None:
            stop = []
        
        response = self.llm(
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            echo=False
        )
        
        return response['choices'][0]['text'].strip()
    
    def __repr__(self) -> str:
        return f"LocalLLM(model_path='{self.model_path}')"

