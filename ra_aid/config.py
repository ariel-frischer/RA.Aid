"""Configuration utilities."""

DEFAULT_RECURSION_LIMIT = 100
DEFAULT_MAX_TEST_CMD_RETRIES = 3
DEFAULT_MAX_TOOL_FAILURES = 3
FALLBACK_TOOL_MODEL_LIMIT = 5
RETRY_FALLBACK_COUNT = 3
DEFAULT_TEST_CMD_TIMEOUT = 60 * 5  # 5 minutes in seconds
DEFAULT_MODEL="claude-3-7-sonnet-20250219"
DEFAULT_SHOW_COST = False

# Context condensation settings
DEFAULT_ENABLE_CONDENSATION = True
DEFAULT_CONDENSATION_QUALITY = "balanced"  # Options: "fast", "balanced", "advanced"
DEFAULT_FAST_CONDENSER_MODEL = "gpt-3.5-turbo"
DEFAULT_BALANCED_CONDENSER_MODEL = "gpt-4o-mini"
DEFAULT_ADVANCED_CONDENSER_MODEL = "deepseek/deepseek-chat-v3-0324"


VALID_PROVIDERS = [
    "anthropic",
    "openai",
    "openrouter",
    "openai-compatible",
    "deepseek",
    "gemini",
    "ollama",
    "fireworks",
    "groq",
]

# Mapping of condensation quality levels to recommended models
CONDENSATION_QUALITY_MODELS = {
    "fast": {
        "model": DEFAULT_FAST_CONDENSER_MODEL,
        "provider": "openai",
        "description": "Fast and cost-effective summarization"
    },
    "balanced": {
        "model": DEFAULT_BALANCED_CONDENSER_MODEL,
        "provider": "openai",
        "description": "Good balance of quality and speed"
    },
    "advanced": {
        "model": DEFAULT_ADVANCED_CONDENSER_MODEL.split("/")[1],
        "provider": DEFAULT_ADVANCED_CONDENSER_MODEL.split("/")[0],
        "description": "High-quality, detailed summarization (slower)"
    }
}
