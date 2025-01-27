"""RetryManager module for handling API retries with exponential backoff."""

import logging
import time
from typing import Callable, Optional, TypeVar

from anthropic import APIError, APITimeoutError, InternalServerError, RateLimitError

from .exceptions import AgentInterrupt
from .logging_config import get_logger

T = TypeVar("T")  # Type variable for the return type


class RetryManager:
    """Handles retry logic with exponential backoff for API calls."""

    def __init__(
        self,
        max_retries: int = 20,
        base_delay: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize RetryManager.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            logger: Optional logger instance
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logger or get_logger(__name__)

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function

        Raises:
            RuntimeError: If max retries exceeded
            KeyboardInterrupt: If interrupted
            AgentInterrupt: If agent interrupted
            Exception: Other unhandled errors
        """
        attempts = 0
        while True:
            try:
                if args or kwargs:
                    return func(*args, **kwargs)
                return func()
            except (KeyboardInterrupt, AgentInterrupt):
                raise
            except (
                InternalServerError,
                APITimeoutError,
                RateLimitError,
                APIError,
                ValueError,
            ) as e:
                if isinstance(e, ValueError):
                    error_str = str(e).lower()
                    if "code" not in error_str or "429" not in error_str:
                        raise  # Re-raise ValueError if it's not a Lambda 429

                if attempts >= self.max_retries:
                    self.logger.error("Max retries reached, failing: %s", str(e))
                    raise RuntimeError(
                        f"Max retries ({self.max_retries}) exceeded. Last error: {e}"
                    )
                attempts += 1

                delay = self.base_delay * (2 ** (attempts - 1))
                self.logger.warning(
                    "API error (attempt %d/%d): %s",
                    attempts,
                    self.max_retries,
                    str(e),
                )

                # Sleep with interrupt checking
                start = time.monotonic()
                while time.monotonic() - start < delay:
                    time.sleep(0.1)
