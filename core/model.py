"""Local LLM model interface for behavior experiments."""

import os
from typing import Optional, List, Dict, Any
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None  # Type hint placeholder
import requests
import json


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
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is not installed. Install with: pip install llama-cpp-python")
        
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
        stop: Optional[list] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate response from the model.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            tools: Optional list of tool definitions (not supported for LocalLLM, kept for compatibility)
            
        Returns:
            Dictionary with 'response' (text) and 'tool_calls' (empty list for LocalLLM)
        """
        if stop is None:
            stop = []
        
        # LocalLLM doesn't support function calling, but we return same format for compatibility
        if tools:
            # Warn that tools are not supported
            pass  # Tools ignored for LocalLLM
        
        response = self.llm(
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            echo=False
        )
        
        return {
            'response': response['choices'][0]['text'].strip(),
            'tool_calls': []
        }
    
    def __repr__(self) -> str:
        return f"LocalLLM(model_path='{self.model_path}')"


class OllamaLLM:
    """Interface for local LLM models using Ollama API."""
    
    @staticmethod
    def list_available_models(base_url: str = "http://localhost:11434") -> List[str]:
        """
        List all available Ollama models.
        
        Args:
            base_url: Base URL for Ollama API
            
        Returns:
            List of available model names
        """
        try:
            response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get('models', [])
            return [m.get('name', '') for m in models if m.get('name')]
        except requests.exceptions.RequestException:
            return []
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama LLM model.
        
        Args:
            model_name: Name of the Ollama model (e.g., 'qwen3:14b')
            base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.model_path = f"ollama:{model_name}"  # For compatibility with runner
        
        # Verify Ollama is accessible
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}: {e}")
        
        # Verify model exists
        models = response.json().get('models', [])
        model_names = [m.get('name', '') for m in models]
        if model_name not in model_names:
            available = ', '.join(model_names[:5])  # Show first 5
            raise ValueError(
                f"Model '{model_name}' not found in Ollama. "
                f"Available models: {available}{'...' if len(model_names) > 5 else ''}"
            )
    
    def infer(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stop: Optional[list] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate response from the model using Ollama API.
        
        Args:
            prompt: Input prompt (or system/user messages if using chat format)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            tools: Optional list of tool definitions in OpenAI format
            
        Returns:
            Dictionary with 'response' (text) and optionally 'tool_calls' (list of function calls)
        """
        if stop is None:
            stop = []
        
        # If tools are provided, use chat API format (supports function calling)
        if tools:
            # Parse prompt into messages if it's a string
            if isinstance(prompt, str):
                # Try to split into system and user prompts
                if "\n\n" in prompt:
                    parts = prompt.split("\n\n", 1)
                    messages = [
                        {"role": "system", "content": parts[0]},
                        {"role": "user", "content": parts[1]}
                    ]
                else:
                    messages = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]
            
            # Prepare chat request with tools
            payload = {
                "model": self.model_name,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                }
            }
            
            if stop:
                payload["options"]["stop"] = stop
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=300  # 5 minutes timeout for long generations
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract response and tool calls
                message = result.get('message', {})
                response_text = message.get('content', '').strip()
                tool_calls = message.get('tool_calls', [])
                
                return {
                    'response': response_text,
                    'tool_calls': tool_calls,
                    'raw_response': result
                }
                
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Error calling Ollama API: {e}")
        
        # No tools - use legacy generate API
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            }
        }
        
        if stop:
            payload["options"]["stop"] = stop
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300  # 5 minutes timeout for long generations
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                'response': result.get('response', '').strip(),
                'tool_calls': []
            }
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Ollama API: {e}")
    
    def __repr__(self) -> str:
        return f"OllamaLLM(model_name='{self.model_name}')"

