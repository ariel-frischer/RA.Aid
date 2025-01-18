"""Unit tests for agent_utils.py."""

import pytest
from unittest.mock import Mock, patch
from langchain_core.language_models import BaseChatModel

from ra_aid.agent_utils import create_agent, get_model_token_limit
from ra_aid.agents.ciayn_agent import CiaynAgent
from ra_aid.models_tokens import models_tokens


@pytest.fixture
def mock_model():
    """Fixture providing a mock LLM model."""
    model = Mock(spec=BaseChatModel)
    return model


@pytest.fixture
def mock_memory():
    """Fixture providing a mock global memory store."""
    with patch('ra_aid.agent_utils._global_memory') as mock_mem:
        mock_mem.get.return_value = {}
        yield mock_mem


def test_get_model_token_limit_anthropic(mock_memory):
    """Test get_model_token_limit with Anthropic model."""
    mock_memory.get.return_value = {
        'provider': 'anthropic',
        'model': 'claude2'
    }
    
    token_limit = get_model_token_limit()
    assert token_limit == models_tokens['anthropic']['claude2']


def test_get_model_token_limit_openai(mock_memory):
    """Test get_model_token_limit with OpenAI model."""
    mock_memory.get.return_value = {
        'provider': 'openai',
        'model': 'gpt-4'
    }
    
    token_limit = get_model_token_limit()
    assert token_limit == models_tokens['openai']['gpt-4']


def test_get_model_token_limit_unknown(mock_memory):
    """Test get_model_token_limit with unknown provider/model."""
    mock_memory.get.return_value = {
        'provider': 'unknown',
        'model': 'unknown-model'
    }
    
    token_limit = get_model_token_limit()
    assert token_limit is None


def test_get_model_token_limit_missing_config(mock_memory):
    """Test get_model_token_limit with missing configuration."""
    mock_memory.get.return_value = {}
    
    token_limit = get_model_token_limit()
    assert token_limit is None


def test_create_agent_anthropic(mock_model, mock_memory):
    """Test create_agent with Anthropic Claude model."""
    mock_memory.get.return_value = {
        'provider': 'anthropic',
        'model': 'claude-2'
    }

    with patch('ra_aid.agent_utils.create_react_agent') as mock_react:
        mock_react.return_value = "react_agent"
        agent = create_agent(mock_model, [])
        
        assert agent == "react_agent"
        mock_react.assert_called_once_with(mock_model, [], checkpointer=None)


def test_create_agent_openai(mock_model, mock_memory):
    """Test create_agent with OpenAI model."""
    mock_memory.get.return_value = {
        'provider': 'openai',
        'model': 'gpt-4'
    }

    with patch('ra_aid.agent_utils.CiaynAgent') as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [])
        
        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(mock_model, [], max_tokens=models_tokens['openai']['gpt-4'])


def test_create_agent_no_token_limit(mock_model, mock_memory):
    """Test create_agent when no token limit is found."""
    mock_memory.get.return_value = {
        'provider': 'unknown',
        'model': 'unknown-model'
    }

    with patch('ra_aid.agent_utils.CiaynAgent') as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [])
        
        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(mock_model, [], max_tokens=None)


def test_create_agent_missing_config(mock_model, mock_memory):
    """Test create_agent with missing configuration."""
    mock_memory.get.return_value = None

    with patch('ra_aid.agent_utils.create_react_agent') as mock_react:
        mock_react.return_value = "react_agent" 
        agent = create_agent(mock_model, [])
        
        assert agent == "react_agent"
        mock_react.assert_called_once_with(mock_model, [], checkpointer=None)


def test_create_agent_error_handling(mock_model, mock_memory):
    """Test create_agent error handling."""
    mock_memory.get.side_effect = Exception("Memory error")

    with patch('ra_aid.agent_utils.create_react_agent') as mock_react:
        mock_react.return_value = "react_agent"
        agent = create_agent(mock_model, [])
        
        assert agent == "react_agent"
        mock_react.assert_called_once_with(mock_model, [], checkpointer=None)


def test_create_agent_with_checkpointer(mock_model, mock_memory):
    """Test create_agent with checkpointer argument."""
    mock_memory.get.return_value = {
        'provider': 'openai',
        'model': 'gpt-4'
    }
    mock_checkpointer = Mock()

    with patch('ra_aid.agent_utils.CiaynAgent') as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [], checkpointer=mock_checkpointer)
        
        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(mock_model, [], max_tokens=models_tokens['openai']['gpt-4'])
