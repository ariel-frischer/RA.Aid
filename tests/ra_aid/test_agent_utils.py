"""Unit tests for agent_utils.py."""

import threading
import time
import pytest
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from unittest.mock import Mock, patch, MagicMock, call
from langchain_core.language_models import BaseChatModel
import litellm
import httpx
from anthropic import APIError, APITimeoutError, RateLimitError, InternalServerError
from ra_aid.agent_utils import RetryManager, TestExecutor, InterruptibleSection
from ra_aid.exceptions import AgentInterrupt

from ra_aid.models_tokens import DEFAULT_TOKEN_LIMIT
from ra_aid.agent_utils import state_modifier, AgentState

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


@pytest.fixture
def mock_retry_manager():
    """Fixture providing a configurable RetryManager."""
    with patch("ra_aid.agent_utils.RetryManager") as mock:
        instance = mock.return_value
        instance.execute = Mock()
        yield instance

@pytest.fixture
def mock_test_executor():
    """Fixture providing a configurable TestExecutor."""
    with patch("ra_aid.agent_utils.TestExecutor") as mock:
        instance = mock.return_value
        instance.execute = Mock(return_value=(True, "prompt", False, 0))
        yield instance

@pytest.fixture
def mock_interruptible_section():
    """Fixture providing a configurable InterruptibleSection."""
    with patch("ra_aid.agent_utils.InterruptibleSection") as mock:
        instance = mock.return_value
        instance.__enter__ = Mock(return_value=instance)
        instance.__exit__ = Mock()
        instance.is_interrupted = Mock(return_value=False)
        yield instance

def test_retry_manager_success():
    """Test RetryManager successfully executes function."""
    retry_manager = RetryManager(max_retries=3, base_delay=0.1)
    mock_func = Mock(return_value="success")
    
    result = retry_manager.execute(mock_func)
    
    assert result == "success"
    mock_func.assert_called_once()

def test_retry_manager_retries_on_api_errors():
    """Test RetryManager retries on different API errors."""
    retry_manager = RetryManager(max_retries=3, base_delay=0.1)
    
    # Test each error type individually
    def test_error(error_class, message, status_code):
        # Create mock response
        response = httpx.Response(
            status_code=status_code,
            text="Error response",  # Basic text response
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )
        
        error_instance = error_class(
            message=message,
            response=response,  # Required httpx.Response
            body={"error": {
                "type": "api_error", 
                "message": message
            }}
        )
        mock_func = Mock(side_effect=[error_instance, "success"])
        result = retry_manager.execute(mock_func)
        assert result == "success"
        assert mock_func.call_count == 2
        mock_func.reset_mock()

    # Test each error type with proper initialization
    test_error(InternalServerError, "Internal server error", 500)
    test_error(APITimeoutError, "Request timed out", 408)
    test_error(RateLimitError, "Rate limit exceeded", 429)
    test_error(APIError, "API request failed", 400)

def test_retry_manager_backoff_delay():
    """Test RetryManager implements exponential backoff."""
    retry_manager = RetryManager(max_retries=3, base_delay=0.1)
    mock_func = Mock(side_effect=[APIError(request="test", message="Error", body="body")] * 2 + ["success"])
    
    start_time = time.time()
    result = retry_manager.execute(mock_func)
    elapsed_time = time.time() - start_time
    
    assert result == "success"
    assert elapsed_time >= 0.3  # 0.1 + 0.2 seconds minimum

def test_retry_manager_interrupt_handling():
    """Test RetryManager handles interrupts properly."""
    retry_manager = RetryManager(max_retries=3, base_delay=0.1)
    mock_func = Mock(side_effect=KeyboardInterrupt)
    
    with pytest.raises(KeyboardInterrupt):
        retry_manager.execute(mock_func)

def test_retry_manager_agent_interrupt():
    """Test RetryManager handles AgentInterrupt properly."""
    retry_manager = RetryManager(max_retries=3, base_delay=0.1)
    mock_func = Mock(side_effect=AgentInterrupt("Test interrupt"))
    
    with pytest.raises(AgentInterrupt):
        retry_manager.execute(mock_func)

def test_test_executor_success():
    """Test TestExecutor handles successful test execution."""
    config = {"test_cmd": "echo test", "auto_test": False}
    executor = TestExecutor(config)
    
    with patch("ra_aid.agent_utils.run_shell_command") as mock_run:
        mock_run.return_value = {"success": True}
        continue_flag, prompt, auto_test, attempts = executor.execute(0, "test prompt")
        
        assert continue_flag
        assert prompt == "test prompt"
        assert not auto_test
        assert attempts == 0

def test_test_executor_failure_with_retry():
    """Test TestExecutor handles test failures with retry."""
    config = {"test_cmd": "false", "auto_test": False}
    executor = TestExecutor(config)
    
    with patch("ra_aid.agent_utils.run_shell_command") as mock_run:
        mock_run.side_effect = Exception("Test failed")
        with patch("ra_aid.agent_utils.ask_human") as mock_ask:
            mock_ask.return_value = "yes"
            continue_flag, prompt, auto_test, attempts = executor.execute(0, "test prompt")
            
            assert not continue_flag
            assert prompt == "test prompt"
            assert attempts == 1

def test_test_executor_auto_test_mode():
    """Test TestExecutor in auto-test mode."""
    config = {"test_cmd": "false", "auto_test": True}
    executor = TestExecutor(config)
    
    with patch("ra_aid.agent_utils.run_shell_command") as mock_run, \
         patch("ra_aid.agent_utils.ask_human") as mock_ask:
        mock_run.side_effect = Exception("Test failed")
        mock_ask.return_value = "auto"
        
        continue_flag, prompt, auto_test, attempts = executor.execute(0, "test prompt")
        
        assert not continue_flag  # Should retry
        assert prompt == "test prompt"  # Prompt unchanged
        assert auto_test  # Auto-test mode enabled
        assert attempts == 1  # One failed attempt
        mock_ask.assert_called()  # Verify human interaction was attempted

def test_interruptible_section_normal():
    """Test InterruptibleSection normal operation."""
    with InterruptibleSection() as section:
        assert not section.is_interrupted()

def test_interruptible_section_thread_safety():
    """Test InterruptibleSection thread safety."""
    results = []
    
    def worker():
        with InterruptibleSection() as section:
            results.append(section.is_interrupted())
    
    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    assert all(not r for r in results)

def test_interruptible_section_cleanup():
    """Test InterruptibleSection proper cleanup."""
    from ra_aid.agent_utils import _CONTEXT_STACK
    
    section = InterruptibleSection()
    with section:
        assert section in _CONTEXT_STACK
    
    assert not section.is_interrupted()
    assert section not in _CONTEXT_STACK

def test_run_agent_with_retry_integration():
    """Test integration of run_agent_with_retry with components."""
    from ra_aid.agent_utils import run_agent_with_retry
    from langchain_core.messages import HumanMessage
    
    mock_agent = Mock()
    mock_agent.stream.return_value = [{"type": "output", "content": "test"}]
    
    with patch("ra_aid.agent_utils.RetryManager") as mock_retry_mgr, \
         patch("ra_aid.agent_utils.TestExecutor") as mock_test_exec, \
         patch("ra_aid.agent_utils.InterruptibleSection") as mock_section, \
         patch("ra_aid.agent_utils._global_memory") as mock_memory:
        
        # Set up mocks
        mock_retry_mgr.return_value.execute.side_effect = lambda f: f()
        mock_test_exec.return_value.execute.return_value = (True, "prompt", False, 0)
        mock_section.return_value.__enter__.return_value.is_interrupted.return_value = False
        mock_memory.get.return_value = 0
        
        result = run_agent_with_retry(mock_agent, "test prompt", {"test_cmd": "echo test"})
        
        # Verify interactions
        assert result == "Agent run completed successfully"
        mock_agent.stream.assert_called_once_with(
            {"messages": [HumanMessage(content="test prompt")]},
            {"test_cmd": "echo test"}
        )


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
