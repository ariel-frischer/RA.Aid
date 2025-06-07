"""
This file contains a sample implementation of the updated CIAYN agent with context condensation support.
This is not the complete implementation, but shows the key changes needed to support context condensation.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from ra_aid.logging_config import get_logger
from ra_aid.models_params import DEFAULT_TOKEN_LIMIT
from ra_aid.context_condensation import (
    ContextCondenser, 
    get_condenser_model_for_quality
)

logger = get_logger(__name__)

class CiaynAgentWithCondensation:
    """Sample implementation of CIAYN agent with context condensation support."""
    
    def __init__(
        self,
        # ... existing parameters ...
        max_tokens: Optional[int] = DEFAULT_TOKEN_LIMIT,
        max_history_messages: int = 100,
        enable_condensation: bool = True,
        condenser_model: Optional[str] = None,
        condenser_provider: Optional[str] = None,
        condensation_quality: str = "balanced",
    ):
        """Initialize the CIAYN agent with condensation support.
        
        Args:
            ... existing parameters ...
            max_tokens: Maximum number of tokens allowed in message history
            max_history_messages: Maximum number of messages to keep in history
            enable_condensation: Whether to enable context condensation
            condenser_model: Model to use for condensing context
            condenser_provider: Provider for the condenser model
            condensation_quality: Quality level for condensation ("fast", "balanced", or "advanced")
        """
        # ... existing initialization ...
        
        self.max_tokens = max_tokens
        self.max_history_messages = max_history_messages
        self.enable_condensation = enable_condensation
        self.condenser_model = condenser_model
        self.condenser_provider = condenser_provider
        self.condensation_quality = condensation_quality
        
        # Initialize condenser if needed
        if self.enable_condensation:
            model, provider = get_condenser_model_for_quality(
                quality=self.condensation_quality,
                custom_model=self.condenser_model,
                custom_provider=self.condenser_provider,
            )
            self.condenser_model = model
            self.condenser_provider = provider
            
    async def _trim_chat_history_with_condensation(
        self,
        initial_messages: List[Any],
        chat_history: List[Any],
    ) -> List[Any]:
        """Trim chat history with context condensation.
        
        Args:
            initial_messages: List of initial messages to preserve
            chat_history: List of chat messages that may be trimmed
            
        Returns:
            List[Any]: Condensed chat history
        """
        # First apply message count limit
        if len(chat_history) > self.max_history_messages:
            chat_history = chat_history[-self.max_history_messages:]
            
        # Skip token limiting if max_tokens is None
        if self.max_tokens is None:
            return initial_messages + chat_history
            
        # Calculate total tokens
        all_messages = initial_messages + chat_history
        total_tokens = sum(self._estimate_tokens(msg) for msg in all_messages)
        
        # If under token limit, no need to condense
        if total_tokens <= self.max_tokens:
            return all_messages
            
        # If condensation is disabled, use traditional trimming
        if not self.enable_condensation:
            return self._trim_chat_history(initial_messages, chat_history)
            
        # Create token counter function
        def token_counter(msgs):
            return sum(self._estimate_tokens(msg) for msg in msgs)
            
        # Use context condenser
        condenser = ContextCondenser(
            condenser_model=self.condenser_model,
            condenser_provider=self.condenser_provider,
        )
        
        # Always keep initial messages
        num_to_keep = len(initial_messages) + 2  # Keep initial messages plus 2 most recent
        
        try:
            # Set a timeout for condensation
            condensed_messages = await asyncio.wait_for(
                condenser.condense_messages(
                    all_messages,
                    token_counter=token_counter,
                    max_tokens=self.max_tokens,
                    num_messages_to_keep=num_to_keep,
                    preserve_system_messages=True,
                ),
                timeout=10.0  # 10 second timeout
            )
            
            logger.debug(
                f"Context condensation: {len(all_messages)} messages → {len(condensed_messages)} messages"
            )
            
            return condensed_messages
            
        except asyncio.TimeoutError:
            logger.warning("Context condensation timed out, falling back to traditional trimming")
            return self._trim_chat_history(initial_messages, chat_history)
        except Exception as e:
            logger.error(f"Error during context condensation: {e}", exc_info=True)
            return self._trim_chat_history(initial_messages, chat_history)
            
    def _trim_chat_history(
        self,
        initial_messages: List[Any],
        chat_history: List[Any],
    ) -> List[Any]:
        """Traditional method to trim chat history based on message count and token limits.
        
        Args:
            initial_messages: List of initial messages to preserve
            chat_history: List of chat messages that may be trimmed
            
        Returns:
            List[Any]: Trimmed chat history
        """
        # First apply message count limit
        if len(chat_history) > self.max_history_messages:
            chat_history = chat_history[-self.max_history_messages:]
            
        # Skip token limiting if max_tokens is None
        if self.max_tokens is None:
            return initial_messages + chat_history
            
        # Calculate initial messages token count
        initial_tokens = sum(self._estimate_tokens(msg) for msg in initial_messages)
        
        # Remove messages from start of chat_history until under token limit
        while chat_history:
            total_tokens = initial_tokens + sum(
                self._estimate_tokens(msg) for msg in chat_history
            )
            if total_tokens <= self.max_tokens:
                break
            chat_history.pop(0)
            
        return initial_messages + chat_history
        
    @staticmethod
    def _estimate_tokens(content: Optional[Union[str, BaseMessage]]) -> int:
        """Estimate token count for a message or string."""
        # ... existing implementation ...
        return 0  # Placeholder
        
    async def _run_agent_iteration(self, *args, **kwargs):
        """Run a single iteration of the agent with context condensation support."""
        # ... existing implementation ...
        
        # Replace the call to _trim_chat_history with _trim_chat_history_with_condensation
        # messages = self._trim_chat_history(initial_messages, chat_history)
        messages = await self._trim_chat_history_with_condensation(initial_messages, chat_history)
        
        # ... rest of the implementation ...