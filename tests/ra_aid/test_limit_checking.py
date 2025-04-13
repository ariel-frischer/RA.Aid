import pytest
from unittest import mock
from decimal import Decimal
import sys

# Assuming the check is implemented within _run_agent_stream or a helper called by it.
# Since _run_agent_stream itself is complex, we'll mock its dependencies heavily.
from ra_aid.agent_utils import _run_agent_stream
from ra_aid.callbacks.default_callback_handler import DefaultCallbackHandler
from ra_aid.database.repositories.config_repository import ConfigRepository
from langgraph.prebuilt.chat_agent_executor import AgentState # Import AgentState

# Fixture to mock ConfigRepository consistently
@pytest.fixture
def mock_config_repo(mocker):
    mock_repo = mocker.MagicMock(spec=ConfigRepository)
    mocker.patch('ra_aid.agent_utils.get_config_repository', return_value=mock_repo)
    # Default behavior: no limits
    mock_repo.get.side_effect = lambda key, default: -1 if key == 'max_tokens' else -1.0 if key == 'max_cost' else default
    return mock_repo

# Fixture to mock DefaultCallbackHandler consistently
@pytest.fixture
def mock_callback_handler(mocker):
    mock_handler_instance = mocker.MagicMock(spec=DefaultCallbackHandler)
    # Default behavior: low usage
    mock_handler_instance.session_totals = {'tokens': 10, 'cost': Decimal('0.001')}
    mocker.patch('ra_aid.agent_utils.DefaultCallbackHandler', return_value=mock_handler_instance)
    # If _run_agent_stream uses the singleton directly via DefaultCallbackHandler()
    mocker.patch('ra_aid.callbacks.default_callback_handler.DefaultCallbackHandler', return_value=mock_handler_instance)
    # Mock the initialization function as well, as it might be called inside _run_agent_stream
    mocker.patch('ra_aid.agent_utils.initialize_callback_handler', return_value=(mock_handler_instance, {}))
    return mock_handler_instance

# Fixture to mock the agent object behavior (stream and state)
@pytest.fixture
def mock_agent(mocker):
    mock_agent_obj = mocker.MagicMock()
    # Simulate the stream yielding one item and then stopping
    mock_agent_obj.stream.return_value = iter([{'type': 'message', 'content': 'test chunk'}])
    # Simulate get_state returning a state with no next step to prevent looping
    mock_state = mocker.MagicMock(spec=AgentState)
    mock_state.next = None # Ensure the outer while loop in _run_agent_stream exits
    mock_agent_obj.get_state.return_value = mock_state
    # Mock invoke just in case, although it shouldn't be called if state.next is None
    mock_agent_obj.invoke.return_value = None
    return mock_agent_obj

# Mocking is_completed and should_exit to control the loop for testing
@pytest.fixture
def mock_loop_controls(mocker):
    # Let the loop run once for the check, then stop
    # is_completed should return False initially, then True to stop *if* reached inside the loop
    # We rely more on the stream ending and state.next being None now
    mocker.patch('ra_aid.agent_utils.is_completed', return_value=False)
    mocker.patch('ra_aid.agent_utils.should_exit', return_value=False)


@mock.patch('ra_aid.agent_utils.sys.exit')
@mock.patch('ra_aid.agent_utils.cpm')
def test_no_limits_set(mock_cpm, mock_sys_exit, mock_config_repo, mock_callback_handler, mock_agent, mock_loop_controls):
    """ Test that execution proceeds normally when no limits are set. """
    # Arrange (Handled by fixtures with default settings)

    # Act
    _run_agent_stream(mock_agent, [mock.Mock()]) # Pass a dummy message list

    # Assert
    mock_sys_exit.assert_not_called()
    # CPM might be called for other reasons (like printing output), so no specific check here unless needed.


@mock.patch('ra_aid.agent_utils.sys.exit')
@mock.patch('ra_aid.agent_utils.cpm')
def test_token_limit_exceeded(mock_cpm, mock_sys_exit, mock_config_repo, mock_callback_handler, mock_agent, mock_loop_controls):
    """ Test that execution stops when token limit is exceeded. """
    # Arrange
    limit_tokens = 50
    current_tokens = 100
    mock_config_repo.get.side_effect = lambda key, default: limit_tokens if key == 'max_tokens' else -1.0 if key == 'max_cost' else default
    mock_callback_handler.session_totals = {'tokens': current_tokens, 'cost': Decimal('0.01')}
    # Mock sys.exit to raise its argument to easily catch it with pytest.raises
    mock_sys_exit.side_effect = SystemExit(1) # Pass exit code 1

    # Act & Assert
    with pytest.raises(SystemExit) as e:
        _run_agent_stream(mock_agent, [mock.Mock()]) # Pass a dummy message list

    # Assert Exit Code
    assert e.value.code == 1
    # Assert Exit Call
    mock_sys_exit.assert_called_once_with(1)
    # Assert Message Print
    mock_cpm.assert_called_once()
    assert f"Execution stopped: Maximum token limit of {limit_tokens} reached" in mock_cpm.call_args[0][0]
    assert f"Session total: {current_tokens}" in mock_cpm.call_args[0][0]


@mock.patch('ra_aid.agent_utils.sys.exit')
@mock.patch('ra_aid.agent_utils.cpm')
def test_cost_limit_exceeded(mock_cpm, mock_sys_exit, mock_config_repo, mock_callback_handler, mock_agent, mock_loop_controls):
    """ Test that execution stops when cost limit is exceeded. """
    # Arrange
    limit_cost_float = 0.05
    limit_cost_decimal = Decimal(str(limit_cost_float))
    current_cost = Decimal('0.10')
    mock_config_repo.get.side_effect = lambda key, default: -1 if key == 'max_tokens' else limit_cost_float if key == 'max_cost' else default
    mock_callback_handler.session_totals = {'tokens': 100, 'cost': current_cost}
    # Mock sys.exit to raise its argument
    mock_sys_exit.side_effect = SystemExit(1) # Pass exit code 1

    # Act & Assert
    with pytest.raises(SystemExit) as e:
        _run_agent_stream(mock_agent, [mock.Mock()]) # Pass a dummy message list

    # Assert Exit Code
    assert e.value.code == 1
    # Assert Exit Call
    mock_sys_exit.assert_called_once_with(1)
    # Assert Message Print
    mock_cpm.assert_called_once()
    # Format expected cost strings carefully
    formatted_max_cost = f"{limit_cost_decimal:.4f}".rstrip('0').rstrip('.')
    formatted_session_cost = f"{current_cost:.4f}".rstrip('0').rstrip('.')
    assert f"Execution stopped: Maximum cost limit of ${formatted_max_cost} reached" in mock_cpm.call_args[0][0]
    assert f"Session total: ${formatted_session_cost}" in mock_cpm.call_args[0][0]


@mock.patch('ra_aid.agent_utils.sys.exit')
@mock.patch('ra_aid.agent_utils.cpm')
def test_limits_not_exceeded(mock_cpm, mock_sys_exit, mock_config_repo, mock_callback_handler, mock_agent, mock_loop_controls):
    """ Test that execution proceeds normally when limits are set but not exceeded. """
    # Arrange
    limit_tokens = 200
    limit_cost = 0.10
    current_tokens = 100
    current_cost = Decimal('0.05')
    mock_config_repo.get.side_effect = lambda key, default: limit_tokens if key == 'max_tokens' else limit_cost if key == 'max_cost' else default
    mock_callback_handler.session_totals = {'tokens': current_tokens, 'cost': current_cost}

    # Act
    _run_agent_stream(mock_agent, [mock.Mock()]) # Pass a dummy message list

    # Assert
    mock_sys_exit.assert_not_called()


@mock.patch('ra_aid.agent_utils.sys.exit')
@mock.patch('ra_aid.agent_utils.cpm')
def test_token_limit_edge_case_zero(mock_cpm, mock_sys_exit, mock_config_repo, mock_callback_handler, mock_agent, mock_loop_controls):
    """ Test token limit edge case where limit is 0 (should exit immediately if any tokens used). """
    # Arrange
    limit_tokens = 0
    current_tokens = 1 # Even 1 token should trigger exit
    mock_config_repo.get.side_effect = lambda key, default: limit_tokens if key == 'max_tokens' else -1.0 if key == 'max_cost' else default
    mock_callback_handler.session_totals = {'tokens': current_tokens, 'cost': Decimal('0.00')}
    mock_sys_exit.side_effect = SystemExit(1) # Pass exit code 1

    # Act & Assert
    with pytest.raises(SystemExit) as e:
        _run_agent_stream(mock_agent, [mock.Mock()]) # Pass a dummy message list

    # Assert Exit Code
    assert e.value.code == 1
    mock_sys_exit.assert_called_once_with(1)
    mock_cpm.assert_called_once()
    assert f"Execution stopped: Maximum token limit of {limit_tokens} reached" in mock_cpm.call_args[0][0]


@mock.patch('ra_aid.agent_utils.sys.exit')
@mock.patch('ra_aid.agent_utils.cpm')
def test_cost_limit_edge_case_zero(mock_cpm, mock_sys_exit, mock_config_repo, mock_callback_handler, mock_agent, mock_loop_controls):
    """ Test cost limit edge case where limit is 0 (should exit immediately if any cost incurred). """
    # Arrange
    limit_cost_float = 0.0
    limit_cost_decimal = Decimal(str(limit_cost_float))
    current_cost = Decimal('0.0001') # Even tiny cost should trigger exit
    mock_config_repo.get.side_effect = lambda key, default: -1 if key == 'max_tokens' else limit_cost_float if key == 'max_cost' else default
    mock_callback_handler.session_totals = {'tokens': 1, 'cost': current_cost}
    mock_sys_exit.side_effect = SystemExit(1) # Pass exit code 1

    # Act & Assert
    with pytest.raises(SystemExit) as e:
        _run_agent_stream(mock_agent, [mock.Mock()]) # Pass a dummy message list

    # Assert Exit Code
    assert e.value.code == 1
    mock_sys_exit.assert_called_once_with(1)
    mock_cpm.assert_called_once()
    formatted_max_cost = f"{limit_cost_decimal:.4f}".rstrip('0').rstrip('.') # Should format to "0"
    assert f"Execution stopped: Maximum cost limit of ${formatted_max_cost} reached" in mock_cpm.call_args[0][0]

