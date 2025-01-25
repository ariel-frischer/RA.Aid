import pytest
from unittest.mock import patch, MagicMock
from ra_aid.tools.shell import run_shell_command
from ra_aid.tools.memory import _global_memory

@pytest.fixture
def mock_console():
    with patch('ra_aid.tools.shell.console') as mock:
        yield mock

@pytest.fixture
def mock_prompt():
    with patch('ra_aid.tools.shell.Prompt') as mock:
        yield mock

@pytest.fixture
def mock_run_interactive():
    with patch('ra_aid.tools.shell.run_interactive_command') as mock:
        mock.return_value = (b"test output", 0)
        yield mock

def test_shell_command_cowboy_mode(mock_console, mock_prompt, mock_run_interactive):
    """Test shell command execution in cowboy mode (no approval)"""
    _global_memory['config'] = {'cowboy_mode': True}
    
    result = run_shell_command.invoke({"command": "echo test"})
    
    assert result['success'] is True
    assert result['return_code'] == 0
    assert "test output" in result['output']
    mock_prompt.ask.assert_not_called()

def test_shell_command_cowboy_message(mock_console, mock_prompt, mock_run_interactive):
    """Test that cowboy mode displays a properly formatted cowboy message with correct spacing"""
    _global_memory['config'] = {'cowboy_mode': True}
    
    with patch('ra_aid.tools.shell.get_cowboy_message') as mock_get_message:
        mock_get_message.return_value = '🤠 Test cowboy message!'
        result = run_shell_command.invoke({"command": "echo test"})
    
    assert result['success'] is True
    mock_console.print.assert_any_call("")
    mock_console.print.assert_any_call(" 🤠 Test cowboy message!")
    mock_console.print.assert_any_call("")
    mock_get_message.assert_called_once()

def test_shell_command_interactive_approved(mock_console, mock_prompt, mock_run_interactive):
    """Test shell command execution with interactive approval"""
    _global_memory['config'] = {'cowboy_mode': False}
    mock_prompt.ask.return_value = 'y'
    
    result = run_shell_command.invoke({"command": "echo test"})
    
    assert result['success'] is True
    assert result['return_code'] == 0
    assert "test output" in result['output']
    mock_prompt.ask.assert_called_once_with(
        "Execute this command? (y=yes, n=no, c=enable cowboy mode for session)",
        choices=["y", "n", "c"],
        default="y",
        show_choices=True,
        show_default=True
    )

def test_shell_command_interactive_rejected(mock_console, mock_prompt, mock_run_interactive):
    """Test shell command rejection in interactive mode"""
    _global_memory['config'] = {'cowboy_mode': False}
    mock_prompt.ask.return_value = 'n'
    
    result = run_shell_command.invoke({"command": "echo test"})
    
    assert result['success'] is False
    assert result['return_code'] == 1
    assert "cancelled by user" in result['output']
    mock_prompt.ask.assert_called_once_with(
        "Execute this command? (y=yes, n=no, c=enable cowboy mode for session)",
        choices=["y", "n", "c"],
        default="y",
        show_choices=True,
        show_default=True
    )
    mock_run_interactive.assert_not_called()

def test_shell_command_execution_error(mock_console, mock_prompt, mock_run_interactive):
    """Test handling of shell command execution errors"""
    _global_memory['config'] = {'cowboy_mode': True}
    mock_run_interactive.side_effect = Exception("Command failed")
    
    result = run_shell_command.invoke({"command": "invalid command"})
    
    assert result['success'] is False
    assert result['return_code'] == 1
    assert "Command failed" in result['output']


def test_test_command_execution(mock_console, mock_prompt, mock_run_interactive):
    """Test execution of test commands with proper output parsing."""
    _global_memory['config'] = {'cowboy_mode': True}
    mock_run_interactive.return_value = (
        b"===== test session starts =====\n"
        b"collected 10 items\n"
        b"===== 8 passed, 2 failed, 0 error =====",
        1
    )
    
    result = run_shell_command.invoke({"command": "pytest tests/"})
    
    assert result['success'] is False
    assert result['return_code'] == 1
    parsed = parse_test_output(result['output'])
    assert parsed['total'] == 10
    assert parsed['passed'] == 8
    assert parsed['failed'] == 2


def test_test_command_success(mock_console, mock_prompt, mock_run_interactive):
    """Test successful test command execution."""
    _global_memory['config'] = {'cowboy_mode': True}
    mock_run_interactive.return_value = (
        b"===== test session starts =====\n"
        b"collected 5 items\n"
        b"===== 5 passed, 0 failed, 0 error =====",
        0
    )
    
    result = run_shell_command.invoke({"command": "pytest"})
    
    assert result['success'] is True
    assert result['return_code'] == 0
    parsed = parse_test_output(result['output'])
    assert parsed['total'] == 5
    assert parsed['passed'] == 5
    assert parsed['failed'] == 0


def test_test_command_error_details(mock_console, mock_prompt, mock_run_interactive):
    """Test extraction of error details from test output."""
    _global_memory['config'] = {'cowboy_mode': True}
    mock_run_interactive.return_value = (
        b"===== test session starts =====\n"
        b"collected 2 items\n"
        b"____________________ test_failure ____________________\n"
        b"def test_failure():\n"
        b"    assert False\n"
        b"E   assert False\n"
        b"===== 1 passed, 1 failed, 0 error =====",
        1
    )
    
    result = run_shell_command.invoke({"command": "pytest"})
    
    parsed = parse_test_output(result['output'])
    assert "test_failure" in parsed['errors']
    assert parsed['failed'] == 1
