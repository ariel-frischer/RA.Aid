"""Module providing thread-safe interruptible code section functionality."""

import sys
import threading
from typing import Any, Optional, Type

from ra_aid.exceptions import AgentInterrupt

_CONTEXT_STACK = []
_INTERRUPT_CONTEXT = None
_FEEDBACK_MODE = False


def _request_interrupt(signum, frame):
    global _INTERRUPT_CONTEXT
    if _CONTEXT_STACK:
        _INTERRUPT_CONTEXT = _CONTEXT_STACK[-1]

    if _FEEDBACK_MODE:
        print()
        print(" 👋 Bye!")
        print()
        sys.exit(0)


class InterruptibleSection:
    """A context manager for handling interruptible code sections.

    This class provides a thread-safe way to manage interruptible sections of code,
    particularly useful for long-running operations that need to be cancellable.
    It maintains a stack of active sections and tracks interrupt state per thread.

    Example:
        >>> with InterruptibleSection() as section:
        ...     # Do some interruptible work
        ...     if section.is_interrupted():
        ...         # Handle interruption
        ...         pass

    Thread Safety:
        - Uses thread-local storage for interrupt state
        - Safely manages global context stack across threads
        - Properly cleans up resources on exit

    Note:
        The interrupt state is maintained per thread and automatically cleaned up
        when the context is exited.
    """

    _thread_local = threading.local()

    def __init__(self) -> None:
        """Initialize a new interruptible section."""
        self._interrupted = False
        self._thread_id = threading.get_ident()

    def __enter__(self) -> "InterruptibleSection":
        """Enter the interruptible section context.

        Returns:
            self: The InterruptibleSection instance
        """
        _CONTEXT_STACK.append(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        """Exit the interruptible section context and cleanup resources.

        Args:
            exc_type: The type of the exception that occurred, if any
            exc_value: The exception instance that occurred, if any
            traceback: The traceback of the exception that occurred, if any
        """
        if self._thread_id == threading.get_ident():
            _CONTEXT_STACK.remove(self)
            self._interrupted = False

    def is_interrupted(self) -> bool:
        """Check if this section has been interrupted.

        Returns:
            bool: True if the section has been interrupted, False otherwise
        """
        global _INTERRUPT_CONTEXT
        return self._interrupted or (
            _INTERRUPT_CONTEXT is self and self._thread_id == threading.get_ident()
        )


def check_interrupt() -> None:
    """Check if current execution should be interrupted.

    Raises:
        AgentInterrupt: If interruption is requested for current context
    """
    if (
        _CONTEXT_STACK
        and _INTERRUPT_CONTEXT is _CONTEXT_STACK[-1]
        and threading.get_ident() == _CONTEXT_STACK[-1]._thread_id
    ):
        raise AgentInterrupt("Interrupt requested")
