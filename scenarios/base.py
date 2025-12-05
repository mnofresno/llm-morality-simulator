"""Base scenario class for LLM behavior experiments."""

from abc import ABC, abstractmethod
from typing import List, Callable, Dict, Any


class BaseScenario(ABC):
    """Base class for all behavior scenarios."""
    
    def __init__(self, name: str):
        """
        Initialize scenario.
        
        Args:
            name: Unique name for the scenario
        """
        self.name = name
    
    @abstractmethod
    def system_prompt(self) -> str:
        """
        Return the system prompt for this scenario.
        
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    def user_prompt(self) -> str:
        """
        Return the user prompt for this scenario.
        
        Returns:
            User prompt string
        """
        pass
    
    @abstractmethod
    def evaluation_functions(self) -> List[Callable]:
        """
        Return list of evaluation functions to extract decisions from responses.
        
        Returns:
            List of callable functions that take a response string and return evaluation results
        """
        pass
    
    def metadata(self) -> Dict[str, Any]:
        """
        Return scenario metadata.
        
        Returns:
            Dictionary with scenario metadata
        """
        return {
            'name': self.name,
            'class': self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

