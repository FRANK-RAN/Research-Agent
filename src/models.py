from openai import OpenAI as OpenAIClient
import os
from typing import Optional, Dict, Any

# Create a completion response class to mimic llama_index's response format
class CompletionResponse:
    def __init__(self, text: str):
        self.text = text

# Custom OpenAI wrapper that follows the llama_index interface
class CustomOpenAI:
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAIClient(api_key=self.api_key)
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        Complete a prompt and return the response in llama_index format.
        
        Args:
            prompt: The prompt text
            
        Returns:
            CompletionResponse with text attribute containing the response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return CompletionResponse(response.choices[0].message.content)
        except Exception as e:
            print(f"[CUSTOM OPENAI] Error calling OpenAI API: {str(e)}")
            return CompletionResponse(f"Error: {str(e)}")