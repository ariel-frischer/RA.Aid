"""Tests for AgentRunner class."""

import signal
import threading
from unittest import mock
from unittest.mock import Mock, patch

import pytest

from ra_aid.agent_runner import AgentRunner
from ra_aid.exceptions import AgentInterrupt
from ra_aid.interruptible_section import InterruptibleSection
from ra_aid.retry_manager import RetryManager
from ra_aid.test_executor import TestExecutor


@pytest.fixture
def mock_agent():
    """Fixture providing a mock agent."""
    agent = Mock()
    agent.stream.return_value = [{"type": "output", "content": "test output"}]
    return agent


@pytest.fixture
def mock_retry_manager():
    """Fixture providing a mock RetryManager."""
    return Mock(spec=RetryManager)


@pytest.fixture
def mock_test_executor():
    """Fixture providing a mock TestExecutor."""
    executor = Mock(spec=TestExecutor)
    executor.execute.return_value = (True, "test prompt", False, 1)
    return executor


@pytest.fixture
def agent_runner(mock_agent, mock_retry_manager, mock_test_executor):
    """Fixture providing a configured AgentRunner instance."""
    config = {"test_key": "test_value"}
    return AgentRunner(
        mock_agent,
        "test prompt",
        config,
        retry_manager=mock_retry_manager,
        test_executor=mock_test_executor,
    )


def test_agent_runner_initialization(agent_runner):
    """Test AgentRunner initializes with correct attributes."""
    assert agent_runner.original_prompt == "test prompt"
    assert agent_runner.current_prompt == "test prompt"
    assert agent_runner.config == {"test_key": "test_value"}
    assert agent_runner.agent_depth == 0
    assert isinstance(agent_runner._interrupt_section, InterruptibleSection)


def test_agent_runner_successful_execution(agent_runner, mock_retry_manager, mock_test_executor):
    """Test successful agent execution flow."""
    # Configure test executor to indicate success
    mock_test_executor.execute.return_value = (True, "test prompt", True, 1)
    mock_retry_manager.execute.return_value = True

    result = agent_runner.run()

    assert result == "Agent run completed successfully"
    # Verify test executor was called with empty prompt
    mock_test_executor.execute.assert_called_once_with("")
    mock_retry_manager.execute.assert_called_once()


def test_agent_runner_with_test_integration(
    agent_runner, mock_test_executor, mock_retry_manager
):
    """Test agent execution with test integration."""
    # Setup test execution flow with initial failure then success
    mock_test_executor.execute.side_effect = [
        (False, "updated prompt with failure", True, 1),  # First call - test fails
        (True, "updated prompt with success", True, 2)    # Second call - test passes
    ]

    # Setup retry manager to return False then True to simulate agent iteration then completion
    mock_retry_manager.execute.side_effect = [False, True]

    result = agent_runner.run()

    assert result == "Agent run completed successfully"

    # Verify test executor was called twice
    assert mock_test_executor.execute.call_count == 2
    mock_test_executor.execute.assert_has_calls([
        mock.call("test prompt"),
        mock.call("")  # Second call during completion check
    ])

    # Verify test executor state was updated correctly
    assert mock_test_executor.auto_test is True
    assert mock_test_executor.test_attempts == 2
    assert agent_runner.current_prompt == "updated prompt with success"

    # Verify retry manager was called twice with correct prompts
    assert mock_retry_manager.execute.call_count == 2
    mock_retry_manager.execute.assert_has_calls([
        mock.call(
            agent_runner._run_agent_iteration,
            agent_runner.agent,
            "updated prompt with failure",
            agent_runner.config,
            agent_runner._interrupt_section
        ),
        mock.call(
            agent_runner._run_agent_iteration,
            agent_runner.agent,
            "updated prompt with success",
            agent_runner.config,
            agent_runner._interrupt_section
        )
    ])


def test_agent_runner_interrupt_handling(agent_runner, mock_retry_manager, mock_test_executor):
    """Test agent handles interrupts properly."""
    # Configure test executor to pass initially
    mock_test_executor.execute.return_value = (True, "test prompt", True, 1)
    
    # Configure retry manager to raise interrupt
    mock_retry_manager.execute.side_effect = AgentInterrupt("Test interrupt")

    with pytest.raises(AgentInterrupt):
        agent_runner.run()

    # Verify test executor was called before interrupt
    mock_test_executor.execute.assert_called_once_with("")


def test_agent_runner_signal_handling(agent_runner):
    """Test signal handling setup and cleanup."""
    original_handler = signal.getsignal(signal.SIGINT)

    with patch("threading.current_thread") as mock_thread:
        mock_thread.return_value = threading.main_thread()
        agent_runner.run()

        # Verify signal handler was restored
        assert signal.getsignal(signal.SIGINT) == original_handler


def test_agent_runner_completion_state(agent_runner, mock_retry_manager):
    """Test completion state handling."""
    with patch(
        "ra_aid.tools.memory._global_memory",
        {"plan_completed": True, "task_completed": False, "completion_message": ""},
    ):
        mock_retry_manager.execute.return_value = True
        result = agent_runner.run()

        assert result == "Agent run completed successfully"


def test_agent_runner_chunk_processing(agent_runner, mock_agent):
    """Test agent output chunk processing."""
    test_chunk = {"type": "output", "content": "test message"}

    with patch("ra_aid.agent_runner.print_agent_output") as mock_print:
        agent_runner._process_chunk(test_chunk)
        mock_print.assert_called_once_with(test_chunk)


def test_agent_runner_resource_cleanup(agent_runner):
    """Test resource cleanup on completion."""
    original_handler = signal.getsignal(signal.SIGINT)

    with patch("threading.current_thread") as mock_thread:
        mock_thread.return_value = threading.main_thread()
        agent_runner.run()

    # Verify signal handler was restored
    current_handler = signal.getsignal(signal.SIGINT)
    assert current_handler == original_handler
