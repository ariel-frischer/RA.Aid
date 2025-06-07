"""Configuration settings for RA.Aid with context condensation support."""

import os
from typing import Dict, Any, Optional, List

# Default model settings
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_PROVIDER = "openai"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_RESEARCH_MODEL = "gpt-4"
DEFAULT_RESEARCH_PROVIDER = "openai"
DEFAULT_RESEARCH_TEMPERATURE = 0.7
DEFAULT_PLANNER_MODEL = "gpt-4"
DEFAULT_PLANNER_PROVIDER = "openai"
DEFAULT_PLANNER_TEMPERATURE = 0.7

# Default token limits
DEFAULT_TOKEN_LIMIT = 8000
DEFAULT_MAX_TOKENS = 1000

# Default cost settings
DEFAULT_MAX_COST = 10.0
DEFAULT_SHOW_COST = True
DEFAULT_EXIT_AT_LIMIT = False

# Default context condensation settings
DEFAULT_ENABLE_CONDENSATION = True
DEFAULT_CONDENSATION_QUALITY = "balanced"  # Options: "fast", "balanced", "advanced"
DEFAULT_CONDENSER_MODEL = "gpt-3.5-turbo"  # Default for "fast" quality
DEFAULT_CONDENSER_PROVIDER = None

# Mapping of condensation quality levels to recommended models
CONDENSATION_QUALITY_MODELS = {
    "fast": {
        "model": "gpt-3.5-turbo",
        "provider": "openai",
        "description": "Fast and cost-effective summarization"
    },
    "balanced": {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "description": "Good balance of quality and speed"
    },
    "advanced": {
        "model": "deepseek-chat-v3-0324",
        "provider": "deepseek",
        "description": "High-quality, detailed summarization (slower)"
    }
}

def get_default_config() -> Dict[str, Any]:
    """Get the default configuration settings.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        # Model settings
        "model": DEFAULT_MODEL,
        "provider": DEFAULT_PROVIDER,
        "temperature": DEFAULT_TEMPERATURE,
        "research_model": DEFAULT_RESEARCH_MODEL,
        "research_provider": DEFAULT_RESEARCH_PROVIDER,
        "research_temperature": DEFAULT_RESEARCH_TEMPERATURE,
        "planner_model": DEFAULT_PLANNER_MODEL,
        "planner_provider": DEFAULT_PLANNER_PROVIDER,
        "planner_temperature": DEFAULT_PLANNER_TEMPERATURE,
        
        # Token limits
        "max_tokens": DEFAULT_MAX_TOKENS,
        "token_limit": DEFAULT_TOKEN_LIMIT,
        
        # Cost settings
        "max_cost": DEFAULT_MAX_COST,
        "show_cost": DEFAULT_SHOW_COST,
        "exit_at_limit": DEFAULT_EXIT_AT_LIMIT,
        
        # Context condensation settings
        "enable_condensation": DEFAULT_ENABLE_CONDENSATION,
        "condensation_quality": DEFAULT_CONDENSATION_QUALITY,
        "condenser_model": DEFAULT_CONDENSER_MODEL,
        "condenser_provider": DEFAULT_CONDENSER_PROVIDER,
    }

def get_condenser_model_for_quality(quality: str) -> Dict[str, str]:
    """Get the recommended condenser model for a quality level.
    
    Args:
        quality: Quality level ("fast", "balanced", or "advanced")
        
    Returns:
        Dict[str, str]: Dictionary with model, provider, and description
    """
    return CONDENSATION_QUALITY_MODELS.get(quality, CONDENSATION_QUALITY_MODELS["balanced"])

def parse_model_string(model_string: str) -> tuple[str, Optional[str]]:
    """Parse a model string that may include a provider prefix.
    
    Args:
        model_string: Model string, possibly with provider prefix (e.g., "openai/gpt-4")
        
    Returns:
        tuple[str, Optional[str]]: (model_name, provider)
    """
    if "/" in model_string:
        provider, model = model_string.split("/", 1)
        return model, provider
    return model_string, None