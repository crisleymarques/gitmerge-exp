from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional

import os

load_dotenv()

PROVIDER_CONFIGS = {
    "google": {
        "model_prefix": "gemini",
        "api_key_env": "GOOGLE_API_KEY",
        "api_base": None
    },
    "groq": {
        "model_prefix": "groq",
        "api_key_env": "GROQ_API_KEY",
        "api_base": None
    },
    "maritaca": {
        "model_prefix": "maritaca",
        "api_key_env": "MARITACA_API_KEY",
        "api_base": "https://api.maritaca.ai/api/v1/chat"
    }
}


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    def get_model_string(self) -> str:
        """
        Returns the model string in the format expected by LiteLLM.
        See: https://docs.litellm.ai/docs/providers
        """
        provider_config = PROVIDER_CONFIGS.get(self.provider, {})
        model_prefix = provider_config.get("model_prefix", self.provider)
        return f"{model_prefix}/{self.model}"


def create_config(provider: str, model: str) -> LLMConfig:
    """
    Create a configuration for a specific provider and model.
    
    Args:
        provider (str): The provider name (google, groq, maritaca)
        model (str): The model name
        
    Returns:
        LLMConfig: Configuration with the appropriate API key
        
    Raises:
        ValueError: If the provider is not supported or API key is missing
    """
    provider_config = PROVIDER_CONFIGS.get(provider)
    if not provider_config:
        raise ValueError(f"Unsupported provider: {provider}")
    
    api_key = os.getenv(provider_config["api_key_env"])
    if not api_key:
        raise ValueError(f"No API key found for provider {provider}")
    
    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=provider_config["api_base"]
    )


def get_default_config() -> LLMConfig:
    """Returns a configuration based on environment variables"""
    provider = os.getenv("DEFAULT_LLM_PROVIDER", "google")
    model = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.0-flash")
    
    return create_config(provider, model) 