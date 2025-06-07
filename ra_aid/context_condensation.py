"""Module for condensing conversation context when token limits are reached."""

import asyncio
from typing import List, Optional, Callable, Dict, Any, Union, Tuple
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
import litellm
from ra_aid.logging_config import get_logger
from ra_aid.model_detection import get_model_name_from_chat_model
from ra_aid.config import (
    DEFAULT_FAST_CONDENSER_MODEL,
    DEFAULT_BALANCED_CONDENSER_MODEL,
    DEFAULT_ADVANCED_CONDENSER_MODEL,
    CONDENSATION_QUALITY_MODELS,
)

logger = get_logger(__name__)

class ContextCondenser:
    """Handles condensation of conversation context when token limits are reached."""
    
    def __init__(
        self,
        condenser_model: str = DEFAULT_FAST_CONDENSER_MODEL,
        condenser_provider: Optional[str] = None,
        max_tokens_per_summary: int = 500,
        summary_buffer_ratio: float = 0.2,  # Reserve 20% of token limit for summaries
        condensation_temperature: float = 0.3,
        condensation_timeout: float = 10.0,  # Timeout in seconds
    ):
        """Initialize the context condenser.
        
        Args:
            condenser_model: Model to use for condensing context
            condenser_provider: Provider for the condenser model
            max_tokens_per_summary: Maximum tokens to use for each summary
            summary_buffer_ratio: Ratio of token limit to reserve for summaries
            condensation_temperature: Temperature for the condensation model
            condensation_timeout: Timeout for condensation requests in seconds
        """
        self.condenser_model = condenser_model
        self.condenser_provider = condenser_provider
        self.max_tokens_per_summary = max_tokens_per_summary
        self.summary_buffer_ratio = summary_buffer_ratio
        self.condensation_temperature = condensation_temperature
        self.condensation_timeout = condensation_timeout
        
    async def condense_messages(
        self,
        messages: List[BaseMessage],
        token_counter: Callable[[List[BaseMessage]], int],
        max_tokens: int,
        num_messages_to_keep: int = 2,
        preserve_system_messages: bool = True,
    ) -> List[BaseMessage]:
        """Condense messages to fit within token limit.
        
        Args:
            messages: List of messages to condense
            token_counter: Function to count tokens
            max_tokens: Maximum tokens allowed
            num_messages_to_keep: Number of most recent messages to always keep
            preserve_system_messages: Whether to always preserve system messages
            
        Returns:
            Condensed list of messages
        """
        if not messages:
            return []
            
        # Check if we're already under the token limit
        current_tokens = token_counter(messages)
        if current_tokens <= max_tokens:
            logger.debug(f"Messages already under token limit ({current_tokens}/{max_tokens}), no condensation needed")
            return messages
            
        logger.info(f"Starting context condensation: {current_tokens} tokens > {max_tokens} limit")
            
        # Identify messages to always keep
        to_keep = []
        to_condense = []
        
        # Always keep system messages if requested
        if preserve_system_messages:
            system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
            to_keep.extend(system_messages)
            
        # Always keep the most recent messages
        recent_messages = messages[-num_messages_to_keep:] if num_messages_to_keep > 0 else []
        to_keep.extend(recent_messages)
        
        # Identify messages to condense
        to_condense = [
            msg for msg in messages 
            if (not preserve_system_messages or not isinstance(msg, SystemMessage)) 
            and msg not in recent_messages
        ]
        
        if not to_condense:
            logger.debug("No messages to condense, returning kept messages")
            return to_keep
            
        # Calculate tokens for messages we're keeping
        kept_tokens = token_counter(to_keep)
        
        # Calculate how many tokens we have available for condensed content
        available_tokens = max_tokens - kept_tokens
        
        # Reserve some tokens for the summary based on the buffer ratio
        summary_tokens = int(max_tokens * self.summary_buffer_ratio)
        available_tokens = max(available_tokens - summary_tokens, 100)  # Ensure at least 100 tokens
        
        logger.debug(f"Token budget: {kept_tokens} (kept) + {summary_tokens} (summary buffer) = {max_tokens - available_tokens}/{max_tokens}")
        
        # Group messages into conversation chunks
        chunks = self._group_messages_into_chunks(to_condense)
        logger.debug(f"Grouped {len(to_condense)} messages into {len(chunks)} chunks for condensation")
        
        # Condense each chunk
        try:
            # Set a timeout for the condensation process
            condensed_chunks = await asyncio.wait_for(
                self._condense_chunks(chunks), 
                timeout=self.condensation_timeout
            )
            
            # Create summary messages
            summary_messages = [
                SystemMessage(content=f"[CONDENSED CONTEXT] {summary}")
                for summary in condensed_chunks
            ]
            
            # Combine everything in the correct order
            # First system messages, then summaries, then recent messages
            system_msgs = [msg for msg in to_keep if isinstance(msg, SystemMessage)]
            non_system_kept = [msg for msg in to_keep if not isinstance(msg, SystemMessage)]
            
            result = system_msgs + summary_messages + non_system_kept
            
            # Final check to ensure we're within token limit
            final_tokens = token_counter(result)
            if final_tokens > max_tokens:
                logger.warning(
                    f"Condensed context still exceeds token limit ({final_tokens}/{max_tokens}). Trimming summaries."
                )
                # Keep system messages and recent messages, drop oldest summaries first
                while summary_messages and token_counter(system_msgs + summary_messages + non_system_kept) > max_tokens:
                    summary_messages.pop(0)
                    
                result = system_msgs + summary_messages + non_system_kept
                
            logger.info(f"Context condensation complete: {len(messages)} messages → {len(result)} messages")
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Context condensation timed out after {self.condensation_timeout}s, falling back to traditional trimming")
            return self._fallback_trimming(messages, token_counter, max_tokens, num_messages_to_keep, preserve_system_messages)
        except Exception as e:
            logger.error(f"Error during context condensation: {e}", exc_info=True)
            return self._fallback_trimming(messages, token_counter, max_tokens, num_messages_to_keep, preserve_system_messages)
    
    def _fallback_trimming(
        self,
        messages: List[BaseMessage],
        token_counter: Callable[[List[BaseMessage]], int],
        max_tokens: int,
        num_messages_to_keep: int = 2,
        preserve_system_messages: bool = True,
    ) -> List[BaseMessage]:
        """Fallback to traditional trimming when condensation fails.
        
        Args:
            messages: List of messages to trim
            token_counter: Function to count tokens
            max_tokens: Maximum tokens allowed
            num_messages_to_keep: Number of most recent messages to always keep
            preserve_system_messages: Whether to always preserve system messages
            
        Returns:
            Trimmed list of messages
        """
        # Always keep system messages if requested
        system_messages = []
        if preserve_system_messages:
            system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
            
        # Always keep the most recent messages
        recent_messages = messages[-num_messages_to_keep:] if num_messages_to_keep > 0 else []
        
        # Calculate tokens for messages we're keeping
        kept_tokens = token_counter(system_messages + recent_messages)
        
        # If we're already over the limit with just the kept messages, keep only system messages and most recent
        if kept_tokens > max_tokens:
            logger.warning(f"Even kept messages exceed token limit ({kept_tokens}/{max_tokens})")
            # Try to keep at least one recent message
            while recent_messages and token_counter(system_messages + recent_messages) > max_tokens:
                recent_messages.pop(0)
            return system_messages + recent_messages
            
        # Find messages we can include (not system or recent)
        other_messages = [
            msg for msg in messages 
            if (not preserve_system_messages or not isinstance(msg, SystemMessage)) 
            and msg not in recent_messages
        ]
        
        # Add as many messages as possible from the end (newest first)
        result = system_messages + recent_messages
        for msg in reversed(other_messages):
            test_result = [msg] + result
            if token_counter(test_result) <= max_tokens:
                result = test_result
            else:
                break
                
        return result
    
    def _group_messages_into_chunks(
        self, 
        messages: List[BaseMessage],
        max_chunk_size: int = 10,
    ) -> List[List[BaseMessage]]:
        """Group messages into logical conversation chunks for summarization.
        
        Args:
            messages: List of messages to group
            max_chunk_size: Maximum number of messages per chunk
            
        Returns:
            List of message chunks
        """
        if not messages:
            return []
            
        # Group messages into conversation chunks
        chunks = []
        current_chunk = []
        
        for msg in messages:
            current_chunk.append(msg)
            
            # Start a new chunk when we reach max size or after an AI message
            if len(current_chunk) >= max_chunk_size or isinstance(msg, AIMessage):
                chunks.append(current_chunk)
                current_chunk = []
                
        # Add any remaining messages
        if current_chunk:
            chunks.append(current_chunk)
            
        # If we have too many chunks, combine smaller ones
        while len(chunks) > 10:
            # Find the smallest chunk
            smallest_idx = min(range(len(chunks)), key=lambda i: len(chunks[i]))
            
            # If it's the last chunk, combine with the previous one
            if smallest_idx == len(chunks) - 1:
                chunks[smallest_idx - 1].extend(chunks[smallest_idx])
                chunks.pop(smallest_idx)
            # Otherwise combine with the next one
            else:
                chunks[smallest_idx].extend(chunks[smallest_idx + 1])
                chunks.pop(smallest_idx + 1)
                
        return chunks
    
    async def _condense_chunks(self, chunks: List[List[BaseMessage]]) -> List[str]:
        """Condense each chunk of messages into a summary.
        
        Args:
            chunks: List of message chunks to condense
            
        Returns:
            List of condensed summaries
        """
        if not chunks:
            return []
            
        # Process chunks in parallel for efficiency
        tasks = [self._condense_chunk(chunk) for chunk in chunks]
        summaries = await asyncio.gather(*tasks)
        
        return summaries
    
    async def _condense_chunk(self, chunk: List[BaseMessage]) -> str:
        """Condense a single chunk of messages into a summary.
        
        Args:
            chunk: List of messages to condense
            
        Returns:
            Condensed summary
        """
        if not chunk:
            return ""
            
        # Convert messages to a format suitable for summarization
        chunk_text = "\n".join([
            f"{msg.type.upper()}: {msg.content}" for msg in chunk
        ])
        
        # Create summarization prompt
        prompt = f"""
        Summarize the following conversation chunk concisely while preserving key information, 
        decisions, and context needed for future reference. Focus on facts, conclusions, and 
        important details that would be needed to continue the conversation effectively.
        
        CONVERSATION CHUNK:
        {chunk_text}
        
        CONCISE SUMMARY:
        """
        
        try:
            # Call the condenser model
            response = await litellm.acompletion(
                model=self.condenser_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens_per_summary,
                temperature=self.condensation_temperature,
                custom_llm_provider=self.condenser_provider,
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            logger.error(f"Error condensing chunk: {e}")
            # Fall back to a simple concatenation
            return f"Conversation about: {chunk[0].content[:100]}..."


def get_condenser_model_for_quality(
    quality: str = "balanced",
    custom_model: Optional[str] = None,
    custom_provider: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Get the appropriate condenser model based on desired quality.
    
    Args:
        quality: Quality level ("fast", "balanced", or "advanced")
        custom_model: Custom model name (overrides quality selection)
        custom_provider: Custom provider name
        
    Returns:
        Tuple[str, Optional[str]]: (model_name, provider)
    """
    if custom_model:
        # If model contains provider prefix, extract it
        if "/" in custom_model and not custom_provider:
            parts = custom_model.split("/", 1)
            provider = parts[0]
            model = parts[1]
            return model, provider
        return custom_model, custom_provider
        
    # Use the quality mapping from config
    quality_model = CONDENSATION_QUALITY_MODELS.get(quality, CONDENSATION_QUALITY_MODELS["balanced"])
    return quality_model["model"], quality_model["provider"]


async def condense_messages_in_state(
    state: Dict[str, Any],
    token_counter: Callable[[List[BaseMessage]], int],
    max_tokens: int,
    condenser_model: str = DEFAULT_FAST_CONDENSER_MODEL,
    condenser_provider: Optional[str] = None,
    enable_condensation: bool = True,
    num_messages_to_keep: int = 2,
    preserve_system_messages: bool = True,
) -> Dict[str, Any]:
    """Condense messages in a state dictionary to fit within token limit.
    
    Args:
        state: State dictionary containing messages
        token_counter: Function to count tokens
        max_tokens: Maximum tokens allowed
        condenser_model: Model to use for condensing context
        condenser_provider: Provider for the condenser model
        enable_condensation: Whether to enable context condensation
        num_messages_to_keep: Number of most recent messages to always keep
        preserve_system_messages: Whether to always preserve system messages
        
    Returns:
        Updated state dictionary with condensed messages
    """
    if not enable_condensation or "messages" not in state:
        return state
        
    messages = state["messages"]
    
    # Create condenser
    condenser = ContextCondenser(
        condenser_model=condenser_model,
        condenser_provider=condenser_provider,
    )
    
    # Condense messages
    condensed_messages = await condenser.condense_messages(
        messages,
        token_counter=token_counter,
        max_tokens=max_tokens,
        num_messages_to_keep=num_messages_to_keep,
        preserve_system_messages=preserve_system_messages,
    )
    
    # Update state
    new_state = state.copy()
    new_state["messages"] = condensed_messages
    
    return new_state