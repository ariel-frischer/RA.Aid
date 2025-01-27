"""Unit tests for test_executor.py and test command execution utilities."""

from unittest.mock import patch

import pytest

from ra_aid.test_executor import TestExecutor, TestState


@pytest.fixture
def test_state():
    """Create a test state fixture."""
    return TestState(prompt="test prompt", test_attempts=0, auto_test=False)


def test_test_executor_success():
    """Test TestExecutor handles successful test execution."""
    config = {"test_cmd": "echo test", "auto_test": False}
    executor = TestExecutor(config)

    with (
        patch("ra_aid.test_executor.run_shell_command") as mock_run,
        patch("ra_aid.test_executor.ask_human") as mock_ask,
    ):
        mock_run.return_value = {"success": True, "output": ""}
        mock_ask.return_value = "y"
        continue_flag, prompt, auto_test, attempts = executor.execute(0, "test prompt")

        assert continue_flag  # Should break since test passed
        assert prompt == "test prompt"
        assert not auto_test
        assert attempts == 1


def test_test_executor_failure_with_retry():
    """Test TestExecutor handles test failures with retry."""
    config = {"test_cmd": "false", "auto_test": False}
    executor = TestExecutor(config)

    with (
        patch("ra_aid.test_executor.run_shell_command") as mock_run,
        patch("ra_aid.test_executor.ask_human") as mock_ask,
    ):
        mock_run.return_value = {"success": False, "output": "error"}
        mock_ask.return_value = "y"
        continue_flag, prompt, auto_test, attempts = executor.execute(0, "test prompt")

        assert not continue_flag  # Should not break on failure
        assert "error" in prompt  # Error should be in prompt
        assert not auto_test
        assert attempts == 1


def test_test_executor_auto_test_mode():
    """Test TestExecutor in auto-test mode."""
    config = {"test_cmd": "false", "auto_test": True}
    executor = TestExecutor(config)

    with patch("ra_aid.test_executor.run_shell_command") as mock_run:
        mock_run.return_value = {"success": False, "output": "error"}

        continue_flag, prompt, auto_test, attempts = executor.execute(0, "test prompt")

        assert not continue_flag  # Should retry
        assert "error" in prompt  # Error should be in prompt
        assert auto_test  # Auto-test mode enabled
        assert attempts == 1  # One failed attempt
        mock_run.assert_called_once()  # Verify test was run


def test_check_max_retries():
    """Test max retries check."""
    config = {"max_test_cmd_retries": 3}
    executor = TestExecutor(config)
    assert not executor.check_max_retries(2)
    assert executor.check_max_retries(3)
    assert executor.check_max_retries(4)


def test_handle_test_failure(test_state):
    """Test handling of test failures."""
    config = {"test_cmd": "test"}
    executor = TestExecutor(config)
    test_result = {"output": "error message"}

    with patch.object(executor, "display_test_failure"):
        state = executor.handle_test_failure(test_state, test_result)
        assert not state.should_break
        assert "error message" in state.prompt


def test_run_test_command_success(test_state):
    """Test successful test command execution."""
    config = {"test_cmd": "test"}
    executor = TestExecutor(config)

    with patch("ra_aid.test_executor.run_shell_command") as mock_run:
        mock_run.return_value = {"success": True, "output": ""}
        state = executor.run_test_command(test_state)
        assert state.should_break
        assert state.test_attempts == 1


def test_run_test_command_failure(test_state):
    """Test failed test command execution."""
    config = {"test_cmd": "test"}
    executor = TestExecutor(config)

    with patch("ra_aid.test_executor.run_shell_command") as mock_run:
        mock_run.return_value = {"success": False, "output": "error"}
        state = executor.run_test_command(test_state)
        assert not state.should_break
        assert state.test_attempts == 1
        assert "error" in state.prompt


def test_run_test_command_error(test_state):
    """Test test command execution error."""
    config = {"test_cmd": "test"}
    executor = TestExecutor(config)

    with patch("ra_aid.test_executor.run_shell_command") as mock_run:
        mock_run.side_effect = Exception("Command failed")
        state = executor.run_test_command(test_state)
        assert state.should_break
        assert state.test_attempts == 1


def test_handle_user_response_no(test_state):
    """Test handling of 'no' response."""
    config = {"test_cmd": "test"}
    executor = TestExecutor(config)
    state = executor.handle_user_response("n", test_state)
    assert state.should_break
    assert not state.auto_test


def test_handle_user_response_auto(test_state):
    """Test handling of 'auto' response."""
    config = {"test_cmd": "test"}
    executor = TestExecutor(config)

    with patch.object(executor, "run_test_command") as mock_run:
        mock_state = TestState("prompt", 1, True, True)
        mock_run.return_value = mock_state
        state = executor.handle_user_response("a", test_state)
        assert state.auto_test
        mock_run.assert_called_once()


def test_handle_user_response_yes(test_state):
    """Test handling of 'yes' response."""
    config = {"test_cmd": "test"}
    executor = TestExecutor(config)

    with patch.object(executor, "run_test_command") as mock_run:
        mock_state = TestState("prompt", 1, False, True)
        mock_run.return_value = mock_state
        state = executor.handle_user_response("y", test_state)
        assert not state.auto_test
        mock_run.assert_called_once()


def test_execute_test_command_no_cmd(test_state):
    """Test execution with no test command."""
    config = {}
    executor = TestExecutor(config)
    result = executor.execute(0, "prompt")
    assert result == (True, "prompt", False, 0)


def test_execute_test_command_manual(test_state):
    """Test manual test execution."""
    config = {"test_cmd": "test"}
    executor = TestExecutor(config)

    with patch("ra_aid.test_executor.ask_human") as mock_ask:
        mock_ask.return_value = "y"
        with patch.object(executor, "run_test_command") as mock_run:
            mock_state = TestState("new prompt", 1, False, True)
            mock_run.return_value = mock_state
            result = executor.execute(0, "prompt")
            assert result == (True, "new prompt", False, 1)
