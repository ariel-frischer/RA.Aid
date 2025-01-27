"""Tests for RetryManager class."""

import time
from unittest.mock import Mock

import httpx
import pytest
from anthropic import APIError, APITimeoutError, InternalServerError, RateLimitError

from ra_aid.exceptions import AgentInterrupt
from ra_aid.retry_manager import RetryManager


@pytest.fixture
def mock_logger():
    """Fixture providing a mock logger."""
    return Mock()


def test_retry_manager_success(mock_logger):
    """Test RetryManager successfully executes function."""
    retry_manager = RetryManager(max_retries=3, base_delay=0.1, logger=mock_logger)
    mock_func = Mock(return_value="success")

    result = retry_manager.execute(mock_func)

    assert result == "success"
    mock_func.assert_called_once()


def test_retry_manager_retries_on_api_errors(mock_logger):
    """Test RetryManager retries on different API errors."""
    retry_manager = RetryManager(max_retries=3, base_delay=0.1, logger=mock_logger)

    # Test different error types
    for error_class in [InternalServerError, APITimeoutError, RateLimitError, APIError]:
        # Create error instance based on error class
        if error_class == APITimeoutError:
            error_instance = error_class(request=Mock(spec=httpx.Request))
        elif error_class == APIError:
            error_instance = error_class(
                "Test error", request=Mock(spec=httpx.Request), body=None
            )
        else:
            mock_response = Mock(spec=httpx.Response)
            mock_response.headers = {}
            mock_response.status_code = 500
            error_instance = error_class(
                "Test error", response=mock_response, body=None
            )
        mock_func = Mock(side_effect=[error_instance, "success"])
        result = retry_manager.execute(mock_func)
        assert result == "success"
        assert mock_func.call_count == 2
        mock_func.reset_mock()


def test_retry_manager_backoff_delay(mock_logger):
    """Test RetryManager implements exponential backoff."""
    retry_manager = RetryManager(max_retries=3, base_delay=0.1, logger=mock_logger)
    mock_func = Mock(
        side_effect=[APIError("Error", request=Mock(spec=httpx.Request), body=None)] * 2
        + ["success"]
    )

    start_time = time.time()
    result = retry_manager.execute(mock_func)
    elapsed_time = time.time() - start_time

    assert result == "success"
    assert elapsed_time >= 0.3  # 0.1 + 0.2 seconds minimum


def test_retry_manager_interrupt_handling(mock_logger):
    """Test RetryManager handles interrupts properly."""
    retry_manager = RetryManager(max_retries=3, base_delay=0.1, logger=mock_logger)

    # Test KeyboardInterrupt
    mock_func = Mock(side_effect=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        retry_manager.execute(mock_func)

    # Test AgentInterrupt
    mock_func = Mock(side_effect=AgentInterrupt("Test interrupt"))
    with pytest.raises(AgentInterrupt):
        retry_manager.execute(mock_func)


def test_retry_manager_max_retries(mock_logger):
    """Test RetryManager raises after max retries."""
    retry_manager = RetryManager(max_retries=2, base_delay=0.1, logger=mock_logger)
    mock_func = Mock(
        side_effect=APIError("Error", request=Mock(spec=httpx.Request), body=None)
    )

    with pytest.raises(RuntimeError) as exc_info:
        retry_manager.execute(mock_func)

    assert "Max retries (2) exceeded" in str(exc_info.value)
    assert mock_func.call_count == 3  # Initial try + 2 retries
