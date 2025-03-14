from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract base class for LLM API providers"""
    
    @abstractmethod
    def generate_notes(self, transcript, verbose=False):
        """Generate notes from transcript using the LLM API"""
        pass