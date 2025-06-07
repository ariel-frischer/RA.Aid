# Context Summarization Implementation Plan

## Overview

This plan outlines the implementation of context summarization for RA.Aid when models reach their token limits. The goal is to condense messages when the context gets too large, rather than simply dropping older messages. This will preserve important information while staying within token limits.

## Current State Analysis

Currently, RA.Aid handles token limits in several ways:

1. **Default Callback Handler**: Tracks token usage and checks against configured limits.
2. **Anthropic Token Limiter**: Provides state modifiers that trim messages when token limits are reached.
3. **CIAYN Agent**: Implements `_trim_chat_history` which removes oldest messages first when token limits are reached.

The current approach is to simply drop older messages when token limits are reached, which can lead to loss of important context.

## Implementation Plan

### 1. Create Context Condensation Module

Create a new module `ra_aid/context_condensation.py` that will handle message summarization:

```python
"""Module for condensing conversation context when token limits are reached."""

from typing import List, Optional, Callable, Dict, Any, Union
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
import litellm
from ra_aid.logging_config import get_logger
from ra_aid.model_detection import get_model_name_from_chat_model

logger = get_logger(__name__)

class ContextCondenser:
    """Handles condensation of conversation context when token limits are reached."""
    
    def __init__(
        self,
        condenser_model: str = "gpt-3.5-turbo",
        condenser_provider: Optional[str] = None,
        max_tokens_per_summary: int = 500,
        summary_buffer_ratio: float = 0.2,  # Reserve 20% of token limit for summaries
    ):
        """Initialize the context condenser.
        
        Args:
            condenser_model: Model to use for condensing context
            condenser_provider: Provider for the condenser model
            max_tokens_per_summary: Maximum tokens to use for each summary
            summary_buffer_ratio: Ratio of token limit to reserve for summaries
        """
        self.condenser_model = condenser_model
        self.condenser_provider = condenser_provider
        self.max_tokens_per_summary = max_tokens_per_summary
        self.summary_buffer_ratio = summary_buffer_ratio
        
    async def condense_messages(
        self,
        messages: List[BaseMessage],
        token_counter: Callable[[List[BaseMessage]], int],
        max_tokens: int,
        num_messages_to_keep: int = 2,
    ) -> List[BaseMessage]:
        """Condense messages to fit within token limit.
        
        Args:
            messages: List of messages to condense
            token_counter: Function to count tokens
            max_tokens: Maximum tokens allowed
            num_messages_to_keep: Number of most recent messages to always keep
            
        Returns:
            Condensed list of messages
        """
        if not messages or token_counter(messages) <= max_tokens:
            return messages
            
        # Always keep system messages and the most recent messages
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        recent_messages = messages[-num_messages_to_keep:]
        
        # Calculate tokens for messages we're keeping
        kept_tokens = token_counter(system_messages + recent_messages)
        
        # Calculate how many tokens we have available for condensed content
        available_tokens = max_tokens - kept_tokens
        
        # Reserve some tokens for the summary
        summary_tokens = int(max_tokens * self.summary_buffer_ratio)
        available_tokens -= summary_tokens
        
        # Messages to condense (excluding system messages and recent messages)
        to_condense = [
            msg for msg in messages 
            if not isinstance(msg, SystemMessage) and msg not in recent_messages
        ]
        
        if not to_condense:
            return system_messages + recent_messages
            
        # Group messages into conversation chunks
        chunks = self._group_messages_into_chunks(to_condense)
        
        # Condense each chunk
        condensed_chunks = await self._condense_chunks(chunks)
        
        # Create summary messages
        summary_messages = [
            SystemMessage(content=f"[CONDENSED CONTEXT] {summary}")
            for summary in condensed_chunks
        ]
        
        # Combine everything
        result = system_messages + summary_messages + recent_messages
        
        # Final check to ensure we're within token limit
        if token_counter(result) > max_tokens:
            # If still over limit, fall back to traditional trimming
            logger.warning(
                "Condensed context still exceeds token limit. Falling back to trimming."
            )
            # Keep system messages and recent messages, drop oldest summaries first
            while summary_messages and token_counter(system_messages + summary_messages + recent_messages) > max_tokens:
                summary_messages.pop(0)
                
            result = system_messages + summary_messages + recent_messages
            
        return result
    
    def _group_messages_into_chunks(self, messages: List[BaseMessage]) -> List[List[BaseMessage]]:
        """Group messages into logical conversation chunks for summarization.
        
        Args:
            messages: List of messages to group
            
        Returns:
            List of message chunks
        """
        # Simple implementation: group by pairs of human and AI messages
        chunks = []
        current_chunk = []
        
        for msg in messages:
            current_chunk.append(msg)
            
            # Start a new chunk after each AI message
            if isinstance(msg, AIMessage) and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                
        # Add any remaining messages
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    async def _condense_chunks(self, chunks: List[List[BaseMessage]]) -> List[str]:
        """Condense each chunk of messages into a summary.
        
        Args:
            chunks: List of message chunks to condense
            
        Returns:
            List of condensed summaries
        """
        summaries = []
        
        for chunk in chunks:
            # Convert messages to a format suitable for summarization
            chunk_text = "\n".join([
                f"{msg.type}: {msg.content}" for msg in chunk
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
                    temperature=0.3,
                    custom_llm_provider=self.condenser_provider,
                )
                
                summary = response.choices[0].message.content.strip()
                summaries.append(summary)
                
            except Exception as e:
                logger.error(f"Error condensing context: {e}")
                # Fall back to a simple concatenation
                summaries.append(f"Conversation about: {chunk_text[:100]}...")
                
        return summaries
```

### 2. Modify Anthropic Token Limiter

Update `ra_aid/anthropic_token_limiter.py` to use the context condenser:

```python
from ra_aid.context_condensation import ContextCondenser

async def condensed_state_modifier(
    state: AgentState, 
    model: BaseChatModel, 
    max_input_tokens: int = DEFAULT_TOKEN_LIMIT,
    condenser_model: str = "gpt-3.5-turbo",
    condenser_provider: Optional[str] = None,
    enable_condensation: bool = True
) -> list[BaseMessage]:
    """Given the agent state and max_tokens, return a condensed list of messages.
    
    This uses context condensation to summarize older messages rather than dropping them.
    
    Args:
        state: The current agent state containing messages
        model: The language model to use for token counting
        max_input_tokens: Maximum number of tokens to allow
        condenser_model: Model to use for condensing context
        condenser_provider: Provider for the condenser model
        enable_condensation: Whether to enable context condensation
        
    Returns:
        list[BaseMessage]: Condensed list of messages that fits within token limit
    """
    messages = state["messages"]
    if not messages:
        return []
        
    model_name = get_model_name_from_chat_model(model)
    wrapped_token_counter = create_token_counter_wrapper(model_name)
    
    # If condensation is disabled, fall back to traditional trimming
    if not enable_condensation:
        return anthropic_trim_messages(
            messages,
            token_counter=wrapped_token_counter,
            max_tokens=max_input_tokens,
            strategy="last",
            allow_partial=False,
            include_system=True,
            num_messages_to_keep=2,
        )
        
    # Use context condenser
    condenser = ContextCondenser(
        condenser_model=condenser_model,
        condenser_provider=condenser_provider,
    )
    
    condensed_messages = await condenser.condense_messages(
        messages,
        token_counter=wrapped_token_counter,
        max_tokens=max_input_tokens,
        num_messages_to_keep=2,
    )
    
    if len(condensed_messages) < len(messages):
        logger.debug(
            f"Context Condenser: {len(messages)} messages → {len(condensed_messages)} messages"
        )
        
    return condensed_messages
```

### 3. Update CIAYN Agent

Modify the `_trim_chat_history` method in `ra_aid/agent_backends/ciayn_agent.py`:

```python
async def _trim_chat_history_with_condensation(
    self,
    initial_messages: List[Any],
    chat_history: List[Any],
    enable_condensation: bool = True,
    condenser_model: str = "gpt-3.5-turbo",
    condenser_provider: Optional[str] = None,
) -> List[Any]:
    """Trim chat history with optional context condensation.
    
    Args:
        initial_messages: List of initial messages to preserve
        chat_history: List of chat messages that may be trimmed
        enable_condensation: Whether to enable context condensation
        condenser_model: Model to use for condensing context
        condenser_provider: Provider for the condenser model
        
    Returns:
        List[Any]: Condensed chat history
    """
    # First apply message count limit
    if len(chat_history) > self.max_history_messages:
        chat_history = chat_history[-self.max_history_messages:]
        
    # Skip token limiting if max_tokens is None
    if self.max_tokens is None:
        return initial_messages + chat_history
        
    # If condensation is disabled, use traditional trimming
    if not enable_condensation:
        return self._trim_chat_history(initial_messages, chat_history)
        
    # Use context condenser
    all_messages = initial_messages + chat_history
    
    # Calculate total tokens
    total_tokens = sum(self._estimate_tokens(msg) for msg in all_messages)
    
    # If under token limit, no need to condense
    if total_tokens <= self.max_tokens:
        return all_messages
        
    # Create token counter function
    def token_counter(msgs):
        return sum(self._estimate_tokens(msg) for msg in msgs)
        
    # Use context condenser
    from ra_aid.context_condensation import ContextCondenser
    
    condenser = ContextCondenser(
        condenser_model=condenser_model,
        condenser_provider=condenser_provider,
    )
    
    # Always keep initial messages
    num_to_keep = len(initial_messages) + 2  # Keep initial messages plus 2 most recent
    
    condensed_messages = await condenser.condense_messages(
        all_messages,
        token_counter=token_counter,
        max_tokens=self.max_tokens,
        num_messages_to_keep=num_to_keep,
    )
    
    return condensed_messages
```

### 4. Update Configuration

Add new configuration options in `ra_aid/config.py`:

```python
# Context condensation settings
DEFAULT_ENABLE_CONDENSATION = True
DEFAULT_CONDENSER_MODEL = "gpt-3.5-turbo"
DEFAULT_CONDENSER_PROVIDER = None
```

### 5. Update CLI Arguments

Add new CLI arguments in `ra_aid/__main__.py`:

```python
# Add to argument parser
parser.add_argument(
    "--enable-condensation",
    action="store_true",
    default=DEFAULT_ENABLE_CONDENSATION,
    help="Enable context condensation when token limits are reached",
)
parser.add_argument(
    "--disable-condensation",
    action="store_true",
    default=False,
    help="Disable context condensation when token limits are reached",
)
parser.add_argument(
    "--condenser-model",
    type=str,
    default=DEFAULT_CONDENSER_MODEL,
    help="Model to use for context condensation",
)
parser.add_argument(
    "--condenser-provider",
    type=str,
    default=DEFAULT_CONDENSER_PROVIDER,
    help="Provider for the condenser model",
)

# Update in the main function
if args.disable_condensation:
    config["enable_condensation"] = False
else:
    config["enable_condensation"] = args.enable_condensation
    
config["condenser_model"] = args.condenser_model
config["condenser_provider"] = args.condenser_provider
```

### 6. Create Prompt Templates

Add new prompt templates in `ra_aid/prompts/condensation_prompts.py`:

```python
"""Prompt templates for context condensation."""

CONDENSATION_PROMPT = """
Summarize the following conversation chunk concisely while preserving key information, 
decisions, and context needed for future reference. Focus on facts, conclusions, and 
important details that would be needed to continue the conversation effectively.

CONVERSATION CHUNK:
{conversation_chunk}

CONCISE SUMMARY:
"""
```

### 7. Integration with ReAct Agent

Update the ReAct agent to support context condensation:

```python
# In the agent creation function
def create_react_agent(
    ...,
    enable_condensation: bool = DEFAULT_ENABLE_CONDENSATION,
    condenser_model: str = DEFAULT_CONDENSER_MODEL,
    condenser_provider: Optional[str] = DEFAULT_CONDENSER_PROVIDER,
    ...
):
    # Configure state transformer with condensation support
    if enable_condensation:
        state_transformer = partial(
            condensed_state_modifier,
            condenser_model=condenser_model,
            condenser_provider=condenser_provider,
            enable_condensation=True,
        )
    else:
        state_transformer = state_modifier
        
    # Use the state transformer in the agent configuration
    ...
```

### 8. Model Selection Logic

Implement logic to select the appropriate condenser model based on the user's preference:

```python
def get_condenser_model(
    config: Dict[str, Any],
    default_model: str = DEFAULT_CONDENSER_MODEL,
) -> Tuple[str, Optional[str]]:
    """Get the condenser model and provider based on configuration.
    
    Args:
        config: Configuration dictionary
        default_model: Default model to use if not specified
        
    Returns:
        Tuple[str, Optional[str]]: (model_name, provider)
    """
    model = config.get("condenser_model", default_model)
    provider = config.get("condenser_provider")
    
    # If model contains provider prefix, extract it
    if "/" in model and not provider:
        parts = model.split("/", 1)
        provider = parts[0]
        model = parts[1]
        
    return model, provider
```

### 9. Testing Plan

1. **Unit Tests**:
   - Test context condenser with various message combinations
   - Test token counting accuracy
   - Test fallback behavior when condensation fails

2. **Integration Tests**:
   - Test with different models and providers
   - Test with and without condensation enabled
   - Test with very large contexts to ensure proper handling

3. **Performance Tests**:
   - Measure latency impact of condensation
   - Compare memory usage with and without condensation

## Implementation Recommendations

1. **Default Models**:
   - Use GPT-3.5 Turbo as the default condenser model for a good balance of speed and quality
   - For users who need higher quality summaries, offer DeepSeek R1 as an advanced option

2. **Provider Selection**:
   - Allow users to specify any provider from the supported list
   - Default to the same provider as the main model if not specified

3. **Hybrid Approach**:
   - For optimal performance, implement a hybrid approach where:
     - First attempt to trim messages using traditional methods
     - Only use condensation when a significant portion of context would be lost
     - Use condensation for important conversation segments only

4. **Graceful Degradation**:
   - If condensation fails or takes too long, fall back to traditional trimming
   - Log warnings when condensation is skipped

## Timeline

1. **Phase 1** (1-2 days):
   - Create the context condensation module
   - Implement basic integration with token limiters

2. **Phase 2** (2-3 days):
   - Update CIAYN and ReAct agents
   - Add configuration options
   - Create unit tests

3. **Phase 3** (1-2 days):
   - Optimize performance
   - Add advanced features (selective condensation)
   - Complete integration tests

4. **Phase 4** (1 day):
   - Documentation
   - Final testing and bug fixes

## Conclusion

This implementation plan provides a comprehensive approach to context summarization in RA.Aid. By condensing older messages rather than dropping them, we can preserve important context while staying within token limits. The plan allows for flexibility in model selection and can be disabled if needed.

The implementation follows the existing architecture patterns in RA.Aid and should integrate seamlessly with the current codebase. The use of async functions ensures that condensation can happen without blocking the main thread, and fallback mechanisms ensure that the system remains robust even if condensation fails.