"""Unit tests for agent_utils.py."""

import pytest
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from unittest.mock import Mock, patch
from langchain_core.language_models import BaseChatModel
import litellm

from ra_aid.models_tokens import DEFAULT_TOKEN_LIMIT
from ra_aid.agent_utils import state_modifier, AgentState, get_effective_model_config

from ra_aid.agent_utils import create_agent, get_model_token_limit
from ra_aid.models_tokens import models_tokens


@pytest.fixture
def mock_model():
    """Fixture providing a mock LLM model."""
    model = Mock(spec=BaseChatModel)
    return model


@pytest.fixture
def mock_memory():
    """Fixture providing a mock global memory store."""
    with patch("ra_aid.agent_utils._global_memory") as mock_mem:
        mock_mem.get.return_value = {}
        yield mock_mem


def test_get_model_token_limit_anthropic(mock_memory):
    """Test get_model_token_limit with Anthropic model."""
    config = {"provider": "anthropic", "model": "claude2"}

    token_limit = get_model_token_limit(config)
    assert token_limit == models_tokens["anthropic"]["claude2"]


def test_get_model_token_limit_openai(mock_memory):
    """Test get_model_token_limit with OpenAI model."""
    config = {"provider": "openai", "model": "gpt-4"}

    token_limit = get_model_token_limit(config)
    assert token_limit == models_tokens["openai"]["gpt-4"]


def test_get_model_token_limit_unknown(mock_memory):
    """Test get_model_token_limit with unknown provider/model."""
    config = {"provider": "unknown", "model": "unknown-model"}

    token_limit = get_model_token_limit(config)
    assert token_limit is None


def test_get_model_token_limit_missing_config(mock_memory):
    """Test get_model_token_limit with missing configuration."""
    config = {}

    token_limit = get_model_token_limit(config)
    assert token_limit is None


@pytest.fixture
def base_config():
    """Base configuration for effective model config tests."""
    return {
        "provider": "anthropic",
        "model": "claude-2",
        "research_provider": "openai",
        "research_model": "gpt-4",
        "planner_provider": "openrouter",
        "expert_model": "claude-3",
    }


def test_full_config_overrides(base_config):
    """Test when both provider and model overrides are provided."""
    result = get_effective_model_config(
        base_config,
        provider_override_key="research_provider",
        model_override_key="research_model",
    )
    assert result["provider"] == "openai"
    assert result["model"] == "gpt-4"


def test_no_overrides_uses_base_values(base_config):
    """Test when no overrides are provided - should use base config."""
    result = get_effective_model_config(base_config)
    assert result["provider"] == "anthropic"
    assert result["model"] == "claude-2"


def test_provider_override_only(base_config):
    """Test when only provider override is specified."""
    result = get_effective_model_config(
        base_config, provider_override_key="planner_provider"
    )
    assert result["provider"] == "openrouter"
    assert result["model"] == "claude-2"  # Falls back to base model


def test_model_override_only(base_config):
    """Test when only model override is specified."""
    result = get_effective_model_config(base_config, model_override_key="expert_model")
    assert result["provider"] == "anthropic"  # Base provider
    assert result["model"] == "claude-3"


def test_null_overrides_fallback_to_base(base_config):
    """Test when override keys exist but have null values."""
    # Set override keys to None
    base_config["research_provider"] = None
    base_config["research_model"] = None

    result = get_effective_model_config(
        base_config,
        provider_override_key="research_provider",
        model_override_key="research_model",
    )
    assert result["provider"] == "anthropic"
    assert result["model"] == "claude-2"


def test_get_effective_model_config_missing_base():
    """Test when base config is missing provider/model but has overrides."""
    config = {"research_provider": "openai", "research_model": "gpt-4"}

    result = get_effective_model_config(
        config,
        provider_override_key="research_provider",
        model_override_key="research_model",
    )

    assert result["provider"] == "openai"
    assert result["model"] == "gpt-4"


def test_get_model_token_limit_litellm_success():
    """Test get_model_token_limit successfully getting limit from litellm."""
    config = {"provider": "anthropic", "model": "claude-2"}

    with patch("ra_aid.agent_utils.get_model_info") as mock_get_info:
        mock_get_info.return_value = {"max_input_tokens": 100000}
        token_limit = get_model_token_limit(config)
        assert token_limit == 100000


def test_get_model_token_limit_litellm_not_found():
    """Test fallback to models_tokens when litellm raises NotFoundError."""
    config = {"provider": "anthropic", "model": "claude-2"}

    with patch("ra_aid.agent_utils.get_model_info") as mock_get_info:
        mock_get_info.side_effect = litellm.exceptions.NotFoundError(
            message="Model not found", model="claude-2", llm_provider="anthropic"
        )
        token_limit = get_model_token_limit(config)
        assert token_limit == models_tokens["anthropic"]["claude2"]


def test_get_model_token_limit_litellm_error():
    """Test fallback to models_tokens when litellm raises other exceptions."""
    config = {"provider": "anthropic", "model": "claude-2"}

    with patch("ra_aid.agent_utils.get_model_info") as mock_get_info:
        mock_get_info.side_effect = Exception("Unknown error")
        token_limit = get_model_token_limit(config)
        assert token_limit == models_tokens["anthropic"]["claude2"]


def test_get_model_token_limit_unexpected_error():
    """Test returning None when unexpected errors occur."""
    config = None  # This will cause an attribute error when accessed

    token_limit = get_model_token_limit(config)
    assert token_limit is None


def test_create_agent_anthropic(mock_model, mock_memory):
    """Test create_agent with Anthropic Claude model."""
    mock_memory.get.return_value = {"provider": "anthropic", "model": "claude-2"}

    with patch("ra_aid.agent_utils.create_react_agent") as mock_react:
        mock_react.return_value = "react_agent"
        agent = create_agent(mock_model, [])

        assert agent == "react_agent"
        mock_react.assert_called_once_with(
            mock_model, [], state_modifier=mock_react.call_args[1]["state_modifier"]
        )


def test_create_agent_openai(mock_model, mock_memory):
    """Test create_agent with OpenAI model."""
    mock_memory.get.return_value = {"provider": "openai", "model": "gpt-4"}

    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [])

        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(
            mock_model, [], max_tokens=models_tokens["openai"]["gpt-4"]
        )


def test_create_agent_no_token_limit(mock_model, mock_memory):
    """Test create_agent when no token limit is found."""
    mock_memory.get.return_value = {"provider": "unknown", "model": "unknown-model"}

    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [])

        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(
            mock_model, [], max_tokens=DEFAULT_TOKEN_LIMIT
        )


def test_create_agent_missing_config(mock_model, mock_memory):
    """Test create_agent with missing configuration."""
    mock_memory.get.return_value = {"provider": "openai"}

    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [])

        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(
            mock_model,
            [],
            max_tokens=DEFAULT_TOKEN_LIMIT,
        )


@pytest.fixture
def mock_messages():
    """Fixture providing mock message objects."""

    return [
        SystemMessage(content="System prompt"),
        HumanMessage(content="Human message 1"),
        AIMessage(content="AI response 1"),
        HumanMessage(content="Human message 2"),
        AIMessage(content="AI response 2"),
    ]


def test_state_modifier(mock_messages):
    """Test that state_modifier correctly trims recent messages while preserving the first message when total tokens > max_tokens."""
    state = AgentState(messages=mock_messages)

    with patch(
        "ra_aid.agents.ciayn_agent.CiaynAgent._estimate_tokens"
    ) as mock_estimate:
        mock_estimate.side_effect = lambda msg: 100 if msg else 0

        result = state_modifier(state, max_input_tokens=250)

        assert len(result) < len(mock_messages)
        assert isinstance(result[0], SystemMessage)
        assert result[-1] == mock_messages[-1]


def test_create_agent_with_checkpointer(mock_model, mock_memory):
    """Test create_agent with checkpointer argument."""
    mock_memory.get.return_value = {"provider": "openai", "model": "gpt-4"}
    mock_checkpointer = Mock()

    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [], checkpointer=mock_checkpointer)

        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(
            mock_model, [], max_tokens=models_tokens["openai"]["gpt-4"]
        )


def test_create_agent_anthropic_token_limiting_enabled(mock_model, mock_memory):
    """Test create_agent sets up token limiting for Claude models when enabled."""
    mock_memory.get.return_value = {
        "provider": "anthropic",
        "model": "claude-2",
        "limit_tokens": True,
    }

    with (
        patch("ra_aid.agent_utils.create_react_agent") as mock_react,
        patch("ra_aid.agent_utils.get_model_token_limit") as mock_limit,
    ):
        mock_react.return_value = "react_agent"
        mock_limit.return_value = 100000

        agent = create_agent(mock_model, [])

        assert agent == "react_agent"
        args = mock_react.call_args
        assert "state_modifier" in args[1]
        assert callable(args[1]["state_modifier"])


def test_research_agent_dedicated_config(mock_model, mock_memory):
    """Test research agent uses dedicated research provider/model when specified."""
    mock_memory.get.return_value = {
        "provider": "anthropic",
        "research_provider": "openai",
        "research_model": "gpt-4",
        "model": "claude-2",
    }

    with patch("ra_aid.agent_utils.create_react_agent") as mock_react:
        mock_react.return_value = "react_agent"
        agent = create_agent(mock_model, [])

        # Verify OpenAI provider is used for research
        assert agent == "react_agent"
        mock_react.assert_called_once()
        # Verify the research config was used for agent creation
        mock_react.assert_called_once()
        # Should use research provider/model but NOT modify global config
        called_model = mock_react.call_args[0][0]
        assert called_model == mock_model


def test_planner_agent_fallback_config(mock_model, mock_memory):
    """Test planner agent falls back to main provider/model when not specified."""
    mock_memory.get.return_value = {
        "provider": "anthropic",
        "model": "claude-2",
        "planner_provider": None,
        "planner_model": None,
    }

    with patch("ra_aid.agent_utils.create_react_agent") as mock_react:
        mock_react.return_value = "react_agent"
        agent = create_agent(mock_model, [])

        # Verify fallback to main provider
        assert agent == "react_agent"
        mock_react.assert_called_once()
        assert mock_memory.get()["provider"] == "anthropic"
        assert mock_memory.get()["model"] == "claude-2"


def test_create_agent_anthropic_token_limiting_disabled(mock_model, mock_memory):
    """Test create_agent doesn't set up token limiting for Claude models when disabled."""
    mock_memory.get.return_value = {
        "provider": "anthropic",
        "model": "claude-2",
        "limit_tokens": False,
    }

    with (
        patch("ra_aid.agent_utils.create_react_agent") as mock_react,
        patch("ra_aid.agent_utils.get_model_token_limit") as mock_limit,
    ):
        mock_react.return_value = "react_agent"
        mock_limit.return_value = 100000

        agent = create_agent(mock_model, [])

        assert agent == "react_agent"
        mock_react.assert_called_once_with(mock_model, [])
