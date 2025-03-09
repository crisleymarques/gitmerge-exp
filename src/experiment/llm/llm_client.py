from litellm import completion
from .llm_config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        
    def generate_content(self, prompt: str) -> str:
        """
        Generate content using the configured LLM provider and model.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            str: The generated response
            
        Raises:
            ValueError: If there's an issue with the configuration
            Exception: For other LLM-related errors
        """
        try:
            response = completion(
                model=self.config.get_model_string(),
                messages=[{"role": "user", "content": prompt}],
                api_key=self.config.api_key,
                api_base=self.config.api_base
            )
            
            return response.choices[0].message.content
            
        except ValueError as e:
            raise ValueError(f"Configuration error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error generating content: {str(e)}")
