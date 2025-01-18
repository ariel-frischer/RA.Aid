"""Unit tests for agent_utils.py."""

from langchain_core.messages import SystemMessage
from ra_aid.models_tokens import DEFAULT_TOKEN_LIMIT
import pytest
from unittest.mock import Mock, patch
from langchain_core.language_models import BaseChatModel

from ra_aid.agent_utils import create_agent, get_model_token_limit, limit_tokens
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
    mock_memory.get.return_value = {"provider": "anthropic", "model": "claude2"}

    token_limit = get_model_token_limit()
    assert token_limit == models_tokens["anthropic"]["claude2"]


def test_get_model_token_limit_openai(mock_memory):
    """Test get_model_token_limit with OpenAI model."""
    mock_memory.get.return_value = {"provider": "openai", "model": "gpt-4"}

    token_limit = get_model_token_limit()
    assert token_limit == models_tokens["openai"]["gpt-4"]


def test_get_model_token_limit_unknown(mock_memory):
    """Test get_model_token_limit with unknown provider/model."""
    mock_memory.get.return_value = {"provider": "unknown", "model": "unknown-model"}

    token_limit = get_model_token_limit()
    assert token_limit is None


def test_get_model_token_limit_missing_config(mock_memory):
    """Test get_model_token_limit with missing configuration."""
    mock_memory.get.return_value = {}

    token_limit = get_model_token_limit()
    assert token_limit is None


def test_create_agent_anthropic(mock_model, mock_memory):
    """Test create_agent with Anthropic Claude model."""
    mock_memory.get.return_value = {"provider": "anthropic", "model": "claude-2"}

    with patch("ra_aid.agent_utils.create_react_agent") as mock_react:
        mock_react.return_value = "react_agent"
        agent = create_agent(mock_model, [])

        assert agent == "react_agent"
        mock_react.assert_called_once_with(mock_model, [])


def test_create_agent_openai(mock_model, mock_memory):
    """Test create_agent with OpenAI model."""
    mock_memory.get.return_value = {"provider": "openai", "model": "gpt-4"}

    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(
            mock_model, [], config={"provider": "openai", "model": "gpt-4"}
        )

        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(
            mock_model,
            [],
            max_tokens=models_tokens["openai"]["gpt-4"]
        )


def test_create_agent_no_token_limit(mock_model, mock_memory):
    """Test create_agent when no token limit is found."""
    mock_memory.get.return_value = {"provider": "unknown", "model": "unknown-model"}

    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [], config={"provider": "other"})

        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(
            mock_model,
            [],
            max_tokens=DEFAULT_TOKEN_LIMIT
        )


def test_create_agent_missing_config(mock_model, mock_memory):
    """Test create_agent with missing configuration."""
    mock_memory.get.return_value = None

    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [], config={"provider": "other"})

        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(
            mock_model,
            [],
            max_tokens=DEFAULT_TOKEN_LIMIT,
        )


@pytest.fixture
def mock_messages():
    """Fixture providing mock message objects."""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    return [
        SystemMessage(content="System prompt"),
        HumanMessage(content="Human message 1"),
        AIMessage(content="AI response 1"),
        HumanMessage(content="Human message 2"),
        AIMessage(content="AI response 2"),
    ]


def test_limit_tokens_basic(mock_messages):
    """Test basic token limiting functionality."""
    with patch(
        "ra_aid.agents.ciayn_agent.CiaynAgent._estimate_tokens"
    ) as mock_estimate:
        # Set up mock to return fixed token counts
        mock_estimate.side_effect = lambda msg: 100 if msg else 0

        result = limit_tokens(mock_messages, max_tokens=250)

        # Should preserve system message and most recent messages
        assert len(result) < len(mock_messages)
        assert isinstance(result[0], SystemMessage)
        assert result[-1] == mock_messages[-1]


def test_limit_tokens_preserves_system(mock_messages):
    """Test that system message is always preserved."""
    with patch(
        "ra_aid.agents.ciayn_agent.CiaynAgent._estimate_tokens"
    ) as mock_estimate:
        mock_estimate.return_value = 1000  # Force aggressive trimming

        result = limit_tokens(mock_messages, max_tokens=100)

        # System message should always be first and preserved
        assert len(result) >= 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "System prompt"


def test_limit_tokens_empty_messages():
    """Test handling of empty message list."""
    result = limit_tokens([], max_tokens=1000)
    assert result == []


def test_limit_tokens_none_messages():
    """Test handling of None messages."""
    result = limit_tokens(None, max_tokens=1000)
    assert result is None


def test_create_agent_error_handling(mock_model, mock_memory):
    """Test create_agent error handling."""
    mock_memory.get.side_effect = Exception("Memory error")

    with patch("ra_aid.agent_utils.create_react_agent") as mock_react:
        mock_react.return_value = "react_agent"
        agent = create_agent(mock_model, [])

        assert agent == "react_agent"
        mock_react.assert_called_once_with(mock_model, [])


def test_create_agent_token_limiting(mock_model, mock_memory):
    """Test create_agent respects token limiting configuration."""
    mock_memory.get.return_value = {"provider": "openai", "model": "gpt-4"}

    # Test with token limiting enabled (default)
    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [], config={})
        
        assert agent == "ciayn_agent"
        assert "max_tokens" in mock_ciayn.call_args[1]

    # Test with token limiting disabled
    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [], config={"limit_tokens": False})
        
        assert agent == "ciayn_agent"
        assert "max_tokens" in mock_ciayn.call_args[1]


def test_create_agent_with_checkpointer(mock_model, mock_memory):
    """Test create_agent with checkpointer argument."""
    mock_memory.get.return_value = {"provider": "openai", "model": "gpt-4"}
    mock_checkpointer = Mock()

    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(
            mock_model,
            [],
            checkpointer=mock_checkpointer,
            config={"provider": "openai", "model": "gpt-4"},
        )

        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(
            mock_model,
            [],
            max_tokens=models_tokens["openai"]["gpt-4"]
        )


def test_create_agent_anthropic_with_token_limit(mock_model, mock_memory):
    """Test create_agent sets up token limiting for Claude models."""
    mock_memory.get.return_value = {"provider": "anthropic", "model": "claude-2"}

    with (
        patch("ra_aid.agent_utils.create_react_agent") as mock_react,
        patch("ra_aid.agent_utils.get_model_token_limit") as mock_limit,
    ):
        mock_react.return_value = "react_agent"
        mock_limit.return_value = 100000

        agent = create_agent(mock_model, [])

        assert agent == "react_agent"
