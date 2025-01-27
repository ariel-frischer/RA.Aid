"""Tests for InterruptibleSection functionality."""

import threading
import pytest
from unittest.mock import patch

from ra_aid.exceptions import AgentInterrupt
from ra_aid.interruptible_section import (
    InterruptibleSection,
    _CONTEXT_STACK,
    check_interrupt,
    _request_interrupt,
)


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
    section = InterruptibleSection()
    with section:
        assert section in _CONTEXT_STACK
    
    assert not section.is_interrupted()
    assert section not in _CONTEXT_STACK


def test_check_interrupt():
    """Test check_interrupt function raises correctly."""
    with InterruptibleSection() as section:
        # Simulate interrupt
        with patch('ra_aid.interruptible_section._INTERRUPT_CONTEXT', section):
            with pytest.raises(AgentInterrupt):
                check_interrupt()


def test_request_interrupt():
    """Test _request_interrupt signal handler."""
    section = InterruptibleSection()
    with section:
        with patch('ra_aid.interruptible_section._FEEDBACK_MODE', False):
            _request_interrupt(None, None)
            assert section.is_interrupted()
