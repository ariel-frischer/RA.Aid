"""Tests for OutputSummarizer class."""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional

from ra_aid.agents.output_summarizer import OutputSummarizer


@pytest.fixture
def mock_redis():
    """Fixture providing a mock Redis client."""
    with patch("redis.Redis") as mock_redis:
        mock_client = Mock()
        mock_redis.return_value = mock_client
        yield mock_client


@pytest.fixture
def basic_summarizer():
    """Fixture providing a basic OutputSummarizer instance."""
    return OutputSummarizer(model="claude-3-haiku-20240307")


@pytest.fixture
def redis_summarizer(mock_redis):
    """Fixture providing an OutputSummarizer instance with Redis enabled."""
    redis_config = {"host": "localhost", "port": 6379, "db": 0}
    return OutputSummarizer(
        model="claude-3-haiku-20240307",
        use_redis=True,
        redis_config=redis_config
    )


def test_summarizer_initialization():
    """Test OutputSummarizer initializes with correct default values."""
    summarizer = OutputSummarizer()
    assert summarizer.model == "claude-3-haiku-20240307"
    assert not summarizer.use_redis
    assert summarizer.redis_config is None


def test_summarizer_with_redis_config(mock_redis):
    """Test OutputSummarizer initializes with Redis configuration."""
    redis_config = {"host": "localhost", "port": 6379, "db": 0}
    summarizer = OutputSummarizer(use_redis=True, redis_config=redis_config)
    
    assert summarizer.use_redis
    assert summarizer.redis_config == redis_config
    mock_redis.assert_called_once_with(**redis_config)


def test_generate_suggestions_basic(basic_summarizer):
    """Test basic suggestion generation for shell output."""
    test_output = "ls: cannot access '/nonexistent': No such file or directory"
    
    with patch("ra_aid.agents.output_summarizer.get_model") as mock_get_model:
        mock_model = Mock()
        mock_model.invoke.return_value.content = "File not found error"
        mock_get_model.return_value = mock_model
        
        suggestions = basic_summarizer._generate_suggestions(test_output)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert "File not found error" in suggestions


def test_generate_suggestions_empty_output(basic_summarizer):
    """Test suggestion generation with empty output."""
    with patch("ra_aid.agents.output_summarizer.get_model") as mock_get_model:
        suggestions = basic_summarizer._generate_suggestions("")
        assert suggestions == []
        mock_get_model.assert_not_called()


def test_cache_operations(redis_summarizer, mock_redis):
    """Test cache operations with Redis."""
    test_key = "test_key"
    test_value = ["suggestion1", "suggestion2"]
    
    # Test cache set
    redis_summarizer._cache_set(test_key, test_value)
    mock_redis.set.assert_called_once()
    
    # Test cache get
    mock_redis.get.return_value = '["suggestion1", "suggestion2"]'
    cached_value = redis_summarizer._cache_get(test_key)
    assert cached_value == test_value
    mock_redis.get.assert_called_once_with(test_key)


def test_cache_operations_no_redis(basic_summarizer):
    """Test cache operations without Redis."""
    test_key = "test_key"
    test_value = ["suggestion1", "suggestion2"]
    
    # Test cache set
    basic_summarizer._cache_set(test_key, test_value)
    assert basic_summarizer._memory_cache.get(test_key) == test_value
    
    # Test cache get
    cached_value = basic_summarizer._cache_get(test_key)
    assert cached_value == test_value


def test_cache_key_generation(basic_summarizer):
    """Test cache key generation for different outputs."""
    output1 = "test output 1"
    output2 = "test output 2"
    
    key1 = basic_summarizer._get_cache_key(output1)
    key2 = basic_summarizer._get_cache_key(output2)
    
    assert key1 != key2
    assert isinstance(key1, str)
    assert len(key1) > 0


def test_error_handling(basic_summarizer):
    """Test error handling in suggestion generation."""
    with patch("ra_aid.agents.output_summarizer.get_model") as mock_get_model:
        mock_model = Mock()
        mock_model.invoke.side_effect = Exception("Model error")
        mock_get_model.return_value = mock_model
        
        suggestions = basic_summarizer._generate_suggestions("test output")
        assert suggestions == []


def test_redis_connection_error():
    """Test handling of Redis connection errors."""
    with patch("redis.Redis") as mock_redis:
        mock_redis.side_effect = Exception("Connection error")
        
        summarizer = OutputSummarizer(
            use_redis=True,
            redis_config={"host": "localhost"}
        )
        
        # Should fallback to memory cache
        assert not summarizer.use_redis
        assert summarizer._memory_cache == {}
