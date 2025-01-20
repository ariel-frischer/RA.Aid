"""Utility functions for working with agents."""

import sys
import time
import uuid
from typing import Literal, Optional, Any, List, Dict, Sequence
from langchain_core.messages import BaseMessage, RemoveMessage, trim_messages

import signal

from langgraph.graph import add_messages
from langgraph.graph.message import Annotated, AnyMessage
from langgraph.prebuilt.chat_agent_executor import AgentState
from ra_aid.models_tokens import DEFAULT_TOKEN_LIMIT, models_tokens
from ra_aid.agents.ciayn_agent import CiaynAgent
import threading

from ra_aid.project_info import (
    get_project_info,
    format_project_info,
    display_project_status,
)

from langgraph.prebuilt import create_react_agent
from ra_aid.console.formatting import print_stage_header, print_error
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from ra_aid.console.output import print_agent_output
from ra_aid.logging_config import get_logger
from ra_aid.exceptions import AgentInterrupt
from ra_aid.tool_configs import (
    get_implementation_tools,
    get_research_tools,
    get_planning_tools,
    get_web_research_tools,
)
from ra_aid.prompts import (
    IMPLEMENTATION_PROMPT,
    EXPERT_PROMPT_SECTION_IMPLEMENTATION,
    HUMAN_PROMPT_SECTION_IMPLEMENTATION,
    EXPERT_PROMPT_SECTION_RESEARCH,
    WEB_RESEARCH_PROMPT_SECTION_RESEARCH,
    WEB_RESEARCH_PROMPT_SECTION_CHAT,
    WEB_RESEARCH_PROMPT_SECTION_PLANNING,
    RESEARCH_PROMPT,
    RESEARCH_ONLY_PROMPT,
    HUMAN_PROMPT_SECTION_RESEARCH,
    PLANNING_PROMPT,
    EXPERT_PROMPT_SECTION_PLANNING,
    HUMAN_PROMPT_SECTION_PLANNING,
    WEB_RESEARCH_PROMPT,
)
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage
from anthropic import APIError, APITimeoutError, RateLimitError, InternalServerError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.tools.memory import (
    _global_memory,
    get_memory_value,
    get_related_files,
)


console = Console()

logger = get_logger(__name__)


@tool
def output_markdown_message(message: str) -> str:
    """Outputs a message to the user, optionally prompting for input."""
    console.print(Panel(Markdown(message.strip()), title="🤖 Assistant"))
    return "Message output."


def estimate_messages_tokens(messages: Sequence[BaseMessage]) -> int:
    """Helper function to estimate total tokens in a sequence of messages.

    Args:
        messages: Sequence of messages to count tokens for

    Returns:
        Total estimated token count
    """
    if not messages:
        return 0

    estimate_tokens = CiaynAgent._estimate_tokens
    return sum(estimate_tokens(msg) for msg in messages)


def trim_messages_with_removal(
    messages: Sequence[BaseMessage], max_tokens: int
) -> List[Dict[str, Any]]:
    """Helper function to trim messages by generating removal operations.

    Args:
        messages: Sequence of messages to trim
        max_tokens: Maximum total tokens allowed

    Returns:
        List of RemoveMessage operations for messages that should be removed
    """
    if not messages:
        return []

    print(f"Total messages: {len(messages)}")

    total_tokens = estimate_messages_tokens(messages)

    print(f"total_tokens={total_tokens}")
    max_tokens = 4000

    # Trim from the front until we fit under max_tokens
    start_idx = 0
    while total_tokens > max_tokens and start_idx < len(token_counts):
        total_tokens -= token_counts[start_idx]
        start_idx += 1

    if start_idx > 0:
        print("=" * 60)
        print("=" * 60)
        print(
            f"TRIMEED {sum(token_counts[:start_idx])} tokens by removing messages 1-{start_idx} out of {len(remaining_msgs)} remaining messages"
        )
        print("=" * 60)
        print("=" * 60)

    filtered_messages = []
    filtered_messages.extend(
        [RemoveMessage(id=msg.id) for msg in remaining_msgs[:start_idx]]
    )
    if initial_message:
        filtered_messages.append(initial_message)
    filtered_messages.extend(remaining_msgs[start_idx:])

    print(
        f"Generated {len(filtered_messages)} total messages ({start_idx} remove operations)"
    )
    return filtered_messages


def limit_tokens(
    state: Sequence[BaseMessage] | Dict[str, Any], max_tokens: int = DEFAULT_TOKEN_LIMIT
) -> Sequence[BaseMessage]:
    """Limit total tokens in state messages while preserving the system message.

    Takes either a LangGraph state list of messages or a dict with messages key and trims
    older messages to stay under max_tokens, ensuring the system (initial) message is preserved.

    Args:
        state: List of messages (BaseMessage) or dict with 'messages' key.
        max_tokens: Maximum total tokens allowed.

    Returns:
        Same type as input (List[BaseMessage] or dict) with messages trimmed to fit under token limit.
    """
    if not state:
        return state

    if isinstance(state, dict):
        messages = state.get("messages", [])
        if not messages:
            return [
                HumanMessage(
                    "Your goal is to add as many files as possible, to try to reach maximum prompt context limit, we are testing your internal messages token limit functionality. If you want to edit a file, you can edit work/SUMMARY.md. Dont make any code changes, just do things that will keep accumulating files and prompt context, fill up any state you can as well."
                )
            ]
    else:
        messages = state

    delete_messages = trim_messages_with_removal(messages, max_tokens)
    return {"messages": delete_messages}


def limit_tokens_remove_style(
    state: AgentState, max_tokens: int = DEFAULT_TOKEN_LIMIT
) -> Dict[str, List[Dict[str, Any]]]:
    """Limit total tokens in state messages while preserving the system message.

    Takes either a LangGraph state list of messages or a dict with messages key and trims
    older messages to stay under max_tokens, ensuring the system (initial) message is preserved.

    Args:
        state: List of messages (BaseMessage) or dict with 'messages' key.
        max_tokens: Maximum total tokens allowed.

    Returns:
        Same type as input (List[BaseMessage] or dict) with messages trimmed to fit under token limit.
    """
    # if not state:
    #     return state

    if isinstance(state, dict):
        messages = state.get("messages", [])
        # print(f"state={state}")
        if not messages:
            print("NOT MESSAGES!")
            return {
                "message": [
                    HumanMessage(
                        "Your goal is to add as many files as possible, to try to reach maximum prompt context limit, we are testing your internal messages token limit functionality. If you want to edit a file, you can edit work/SUMMARY.md. Dont make any code changes, just do things that will keep accumulating files and prompt context, fill up any state you can as well."
                    )
                ]
            }
    else:
        logger.warn("State is not dict")
        messages = state

    filtered_messages = trim_messages_with_removal(messages, max_tokens)
    # filtered_messages = trim_messages(messages, max_tokens)
    # state["messages"] = filtered_messages

    return filtered_messages


def state_modifier(
    state: AgentState, model: BaseChatModel, max_tokens: int = DEFAULT_TOKEN_LIMIT
) -> list[BaseMessage]:
    """Given the agent state, and max_tokens return a trimmed list of messages for the chat model."""

    print(f"max_tokens={max_tokens}")
    max_tokens = 12000
    length = len(state["messages"])
    print(f"len(state[messages]): {length}")
    # tokens = model.get_num_tokens_from_messages(state["messages"])
    tokens = estimate_messages_tokens(state["messages"])
    print(f"tokens={tokens}")
    messages = state["messages"]
    print(f"messages={[(msg.id, msg.type) for msg in messages]}")

    trimmed_messages = trim_messages(
        state["messages"],
        token_counter=estimate_messages_tokens,
        max_tokens=max_tokens,
        strategy="last",
        allow_partial=False,
    )
    print(f"len trimmed_messages={len(trimmed_messages)}")

    # trimmed_tokens = model.get_num_tokens_from_messages(trimmed_messages)
    trimmed_tokens = estimate_messages_tokens(trimmed_messages)
    print(f"trimmed_tokens={trimmed_tokens}")
    print("TRIMMED:")
    print(f"trimmed_messages={[msg.id for msg in trimmed_messages]}")
    # print(trimmed_messages)

    return trimmed_messages


def get_model_token_limit() -> Optional[int]:
    """Get the token limit for the current model configuration.

    Returns:
        Optional[int]: The token limit if found, None otherwise
    """
    try:
        config = _global_memory.get("config") or {}
        provider = config.get("provider")
        model_name = config.get("model")

        # Get token limit for this model
        token_limit = None
        if provider and model_name:
            provider_tokens = models_tokens.get(provider, {})
            token_limit = provider_tokens.get(model_name)
            if token_limit:
                logger.debug(
                    f"Found token limit for {provider}/{model_name}: {token_limit}"
                )
            else:
                logger.debug(f"Could not find token limit for {provider}/{model_name}")
                token_limit = None

        return token_limit

    except Exception as e:
        logger.warning(f"Failed to get model token limit: {e}")
        return None


def my_custom_add_messages(
    left: List[Any],
    right: List[Any],
    *,
    format: Optional[Literal["langchain-openai"]] = None,
) -> List[Any]:
    """
    Similar to `add_messages`, but also handles messages whose `type='remove'`
    by removing them, even if they are not actual `RemoveMessage` objects.
    """

    # 1) Convert to list if either side is not already a list
    if not isinstance(left, list):
        left = [left]  # type: ignore[assignment]
    if not isinstance(right, list):
        right = [right]  # type: ignore[assignment]

    # 2) Convert each entry to a message if needed (pseudo-implementations)
    #    Replace these stubs with whatever "convert_to_messages" does in your code.
    def convert_to_messages(objs: List[Any]) -> List[Any]:
        # < your actual convert_to_messages here >
        return objs

    def message_chunk_to_message(m: Any) -> Any:
        # < your actual message_chunk_to_message here >
        return m

    left = [message_chunk_to_message(m) for m in convert_to_messages(left)]
    right = [message_chunk_to_message(m) for m in convert_to_messages(right)]

    # 3) Ensure each message has an ID
    for m in left:
        if not getattr(m, "id", None):
            m.id = str(uuid.uuid4())
    for m in right:
        if not getattr(m, "id", None):
            m.id = str(uuid.uuid4())

    # 4) Merge logic
    left_idx_by_id = {m.id: i for i, m in enumerate(left)}
    merged = left.copy()
    ids_to_remove = set()

    for m in right:
        # interpret "type='remove'" as instruction to remove
        if getattr(m, "type", None) == "remove":
            existing_idx = left_idx_by_id.get(m.id)
            if existing_idx is not None:
                ids_to_remove.add(m.id)
            else:
                # Decide whether to raise an error or just ignore
                # If you prefer not to raise, comment this out
                raise ValueError(
                    f"Attempt to remove a message with an ID that doesn't exist: {m.id}"
                )
        else:
            # If the ID already exists, we replace
            existing_idx = left_idx_by_id.get(m.id)
            if existing_idx is not None:
                merged[existing_idx] = m
            else:
                # If ID not found, append
                merged.append(m)

    # Filter out any IDs we decided to remove
    merged = [m for m in merged if m.id not in ids_to_remove]

    # 5) Optional format step
    # if format == "langchain-openai":
    #     merged = _format_messages(merged)  # implement or import your method
    # elif format:
    #     msg = f"Unrecognized format: '{format}'. Expected one of 'langchain-openai', None."
    #     raise ValueError(msg)

    return merged


class TokenLimitedState(AgentState):
    # messages: Annotated[list[AnyMessage], add_messages]
    messages: Annotated[list[AnyMessage], my_custom_add_messages]
    # remaining_steps: RemainingSteps
    # is_last_step: IsLastStep


def build_agent_kwargs(
    model: BaseChatModel,
    checkpointer: Optional[Any] = None,
    config: Dict[str, Any] = None,
    token_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Build kwargs dictionary for agent creation.

    Args:
        checkpointer: Optional memory checkpointer
        config: Optional configuration dictionary
        token_limit: Optional token limit for the model

    Returns:
        Dictionary of kwargs for agent creation
    """
    agent_kwargs = {}

    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer

    provider = config.get("provider", "")
    if config.get("limit_tokens", True) and provider.lower() == "anthropic":

        def wrapped_state_modifier(state: AgentState) -> list[BaseMessage]:
            return state_modifier(state, model, max_tokens=token_limit)

        agent_kwargs["state_modifier"] = wrapped_state_modifier
        # agent_kwargs["state_schema"] = TokenLimitedState

    return agent_kwargs


def create_agent(
    model: BaseChatModel,
    tools: List[Any],
    *,
    checkpointer: Any = None,
) -> Any:
    """Create a react agent with the given configuration.

    Args:
        model: The LLM model to use
        tools: List of tools to provide to the agent
        checkpointer: Optional memory checkpointer
        config: Optional configuration dictionary containing settings like:
            - limit_tokens (bool): Whether to apply token limiting (default: True)
            - provider (str): The LLM provider name
            - model (str): The model name

    Returns:
        The created agent instance

    Token limiting helps prevent context window overflow by trimming older messages
    while preserving system messages. It can be disabled by setting
    config['limit_tokens'] = False.
    """
    try:
        config = _global_memory.get("config", {})
        provider = config.get("provider")
        model_name = config.get("model")
        token_limit = get_model_token_limit() or DEFAULT_TOKEN_LIMIT

        # Use REACT agent for Anthropic Claude models, otherwise use CIAYN
        if provider == "anthropic" and model_name and "claude" in model_name:
            logger.debug("Using create_react_agent to instantiate agent.")
            agent_kwargs = build_agent_kwargs(model, checkpointer, config, token_limit)
            # TODO: move to right before adding state modifier

            # model.bind_tools(tools)
            return create_react_agent(model, tools, **agent_kwargs)
        else:
            logger.debug("Using CiaynAgent agent instance")
            return CiaynAgent(model, tools, max_tokens=token_limit)

    except Exception as e:
        # Default to REACT agent if provider/model detection fails
        logger.warning(f"Failed to detect model type: {e}. Defaulting to REACT agent.")
        config = _global_memory.get("config", {})
        token_limit = get_model_token_limit()
        agent_kwargs = build_agent_kwargs(model, checkpointer, config, token_limit)
        return create_react_agent(model, tools, **agent_kwargs)


def run_research_agent(
    base_task_or_query: str,
    model,
    *,
    expert_enabled: bool = False,
    research_only: bool = False,
    hil: bool = False,
    web_research_enabled: bool = False,
    memory: Optional[Any] = None,
    config: Optional[dict] = None,
    thread_id: Optional[str] = None,
    console_message: Optional[str] = None,
) -> Optional[str]:
    """Run a research agent with the given configuration.

    Args:
        base_task_or_query: The main task or query for research
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        research_only: Whether this is a research-only task
        hil: Whether human-in-the-loop mode is enabled
        web_research_enabled: Whether web research is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)
        console_message: Optional message to display before running

    Returns:
        Optional[str]: The completion message if task completed successfully

    Example:
        result = run_research_agent(
            "Research Python async patterns",
            model,
            expert_enabled=True,
            research_only=True
        )
    """
    thread_id = thread_id or str(uuid.uuid4())
    logger.debug("Starting research agent with thread_id=%s", thread_id)
    logger.debug(
        "Research configuration: expert=%s, research_only=%s, hil=%s, web=%s",
        expert_enabled,
        research_only,
        hil,
        web_research_enabled,
    )

    # Initialize memory if not provided
    if memory is None:
        memory = MemorySaver()

    # Set up thread ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Configure tools
    tools = get_research_tools(
        research_only=research_only,
        expert_enabled=expert_enabled,
        human_interaction=hil,
        web_research_enabled=config.get("web_research_enabled", False),
    )

    # Create agent
    agent = create_agent(model, tools, checkpointer=memory)

    # Format prompt sections
    expert_section = EXPERT_PROMPT_SECTION_RESEARCH if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_RESEARCH if hil else ""
    web_research_section = (
        WEB_RESEARCH_PROMPT_SECTION_RESEARCH
        if config.get("web_research_enabled")
        else ""
    )

    # Get research context from memory
    key_facts = _global_memory.get("key_facts", "")
    code_snippets = _global_memory.get("code_snippets", "")
    related_files = _global_memory.get("related_files", "")

    # Get project info
    try:
        project_info = get_project_info(".", file_limit=2000)
        formatted_project_info = format_project_info(project_info)
    except Exception as e:
        logger.warning(f"Failed to get project info: {e}")
        formatted_project_info = ""

    # Build prompt
    prompt = (RESEARCH_ONLY_PROMPT if research_only else RESEARCH_PROMPT).format(
        base_task=base_task_or_query,
        research_only_note=""
        if research_only
        else " Only request implementation if the user explicitly asked for changes to be made.",
        expert_section=expert_section,
        human_section=human_section,
        web_research_section=web_research_section,
        key_facts=key_facts,
        work_log=get_memory_value("work_log"),
        code_snippets=code_snippets,
        related_files=related_files,
        project_info=formatted_project_info,
    )

    # Set up configuration
    run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    if config:
        run_config.update(config)

    try:
        # Display console message if provided
        if console_message:
            console.print(
                Panel(Markdown(console_message), title="🔬 Looking into it...")
            )

        if project_info:
            display_project_status(project_info)

        # Run agent with retry logic if available
        if agent is not None:
            logger.debug("Research agent completed successfully")
            return run_agent_with_retry(agent, prompt, run_config)
        else:
            # Just run web research tools directly if no agent
            logger.debug("No model provided, running web research tools directly")
            return run_web_research_agent(
                base_task_or_query,
                model=None,
                expert_enabled=expert_enabled,
                hil=hil,
                web_research_enabled=web_research_enabled,
                memory=memory,
                config=config,
                thread_id=thread_id,
                console_message=console_message,
            )
    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Research agent failed: %s", str(e), exc_info=True)
        raise


def run_web_research_agent(
    query: str,
    model,
    *,
    expert_enabled: bool = False,
    hil: bool = False,
    web_research_enabled: bool = False,
    memory: Optional[Any] = None,
    config: Optional[dict] = None,
    thread_id: Optional[str] = None,
    console_message: Optional[str] = None,
) -> Optional[str]:
    """Run a web research agent with the given configuration.

    Args:
        query: The mainquery for web research
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        hil: Whether human-in-the-loop mode is enabled
        web_research_enabled: Whether web research is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)
        console_message: Optional message to display before running

    Returns:
        Optional[str]: The completion message if task completed successfully

    Example:
        result = run_web_research_agent(
            "Research latest Python async patterns",
            model,
            expert_enabled=True
        )
    """
    thread_id = thread_id or str(uuid.uuid4())
    logger.debug("Starting web research agent with thread_id=%s", thread_id)
    logger.debug(
        "Web research configuration: expert=%s, hil=%s, web=%s",
        expert_enabled,
        hil,
        web_research_enabled,
    )

    # Initialize memory if not provided
    if memory is None:
        memory = MemorySaver()

    # Set up thread ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Configure tools using restricted web research toolset
    tools = get_web_research_tools(expert_enabled=expert_enabled)

    # Create agent
    agent = create_agent(model, tools, checkpointer=memory)

    # Format prompt sections
    expert_section = EXPERT_PROMPT_SECTION_RESEARCH if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_RESEARCH if hil else ""

    # Get research context from memory
    key_facts = _global_memory.get("key_facts", "")
    code_snippets = _global_memory.get("code_snippets", "")
    related_files = _global_memory.get("related_files", "")

    # Build prompt
    prompt = WEB_RESEARCH_PROMPT.format(
        web_research_query=query,
        expert_section=expert_section,
        human_section=human_section,
        key_facts=key_facts,
        code_snippets=code_snippets,
        related_files=related_files,
    )

    # Set up configuration
    run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    if config:
        run_config.update(config)

    try:
        # Display console message if provided
        if console_message:
            console.print(Panel(Markdown(console_message), title="🔬 Researching..."))

        logger.debug("Web research agent completed successfully")
        return run_agent_with_retry(agent, prompt, run_config)

    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Web research agent failed: %s", str(e), exc_info=True)
        raise


def run_planning_agent(
    base_task: str,
    model,
    *,
    expert_enabled: bool = False,
    hil: bool = False,
    memory: Optional[Any] = None,
    config: Optional[dict] = None,
    thread_id: Optional[str] = None,
) -> Optional[str]:
    """Run a planning agent to create implementation plans.

    Args:
        base_task: The main task to plan implementation for
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        hil: Whether human-in-the-loop mode is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)

    Returns:
        Optional[str]: The completion message if planning completed successfully
    """
    thread_id = thread_id or str(uuid.uuid4())
    logger.debug("Starting planning agent with thread_id=%s", thread_id)
    logger.debug("Planning configuration: expert=%s, hil=%s", expert_enabled, hil)

    # Initialize memory if not provided
    if memory is None:
        memory = MemorySaver()

    # Set up thread ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Configure tools
    tools = get_planning_tools(
        expert_enabled=expert_enabled,
        web_research_enabled=config.get("web_research_enabled", False),
    )

    # Create agent
    agent = create_agent(model, tools, checkpointer=memory)

    # Format prompt sections
    expert_section = EXPERT_PROMPT_SECTION_PLANNING if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_PLANNING if hil else ""
    web_research_section = (
        WEB_RESEARCH_PROMPT_SECTION_PLANNING
        if config.get("web_research_enabled")
        else ""
    )

    # Build prompt
    planning_prompt = PLANNING_PROMPT.format(
        expert_section=expert_section,
        human_section=human_section,
        web_research_section=web_research_section,
        base_task=base_task,
        research_notes=get_memory_value("research_notes"),
        related_files="\n".join(get_related_files()),
        key_facts=get_memory_value("key_facts"),
        key_snippets=get_memory_value("key_snippets"),
        work_log=get_memory_value("work_log"),
        research_only_note=""
        if config.get("research_only")
        else " Only request implementation if the user explicitly asked for changes to be made.",
    )

    # Set up configuration
    run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    if config:
        run_config.update(config)

    try:
        print_stage_header("Planning Stage")
        logger.debug("Planning agent completed successfully")
        return run_agent_with_retry(agent, planning_prompt, run_config)
    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Planning agent failed: %s", str(e), exc_info=True)
        raise


def run_task_implementation_agent(
    base_task: str,
    tasks: list,
    task: str,
    plan: str,
    related_files: list,
    model,
    *,
    expert_enabled: bool = False,
    web_research_enabled: bool = False,
    memory: Optional[Any] = None,
    config: Optional[dict] = None,
    thread_id: Optional[str] = None,
) -> Optional[str]:
    """Run an implementation agent for a specific task.

    Args:
        base_task: The main task being implemented
        tasks: List of tasks to implement
        plan: The implementation plan
        related_files: List of related files
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        web_research_enabled: Whether web research is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)

    Returns:
        Optional[str]: The completion message if task completed successfully
    """
    thread_id = thread_id or str(uuid.uuid4())
    logger.debug("Starting implementation agent with thread_id=%s", thread_id)
    logger.debug(
        "Implementation configuration: expert=%s, web=%s",
        expert_enabled,
        web_research_enabled,
    )
    logger.debug("Task details: base_task=%s, current_task=%s", base_task, task)
    logger.debug("Related files: %s", related_files)

    # Initialize memory if not provided
    if memory is None:
        memory = MemorySaver()

    # Set up thread ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Configure tools
    tools = get_implementation_tools(
        expert_enabled=expert_enabled,
        web_research_enabled=config.get("web_research_enabled", False),
    )

    # Create agent
    agent = create_agent(model, tools, checkpointer=memory)

    # Build prompt
    prompt = IMPLEMENTATION_PROMPT.format(
        base_task=base_task,
        task=task,
        tasks=tasks,
        plan=plan,
        related_files=related_files,
        key_facts=get_memory_value("key_facts"),
        key_snippets=get_memory_value("key_snippets"),
        research_notes=get_memory_value("research_notes"),
        work_log=get_memory_value("work_log"),
        expert_section=EXPERT_PROMPT_SECTION_IMPLEMENTATION if expert_enabled else "",
        human_section=HUMAN_PROMPT_SECTION_IMPLEMENTATION
        if _global_memory.get("config", {}).get("hil", False)
        else "",
        web_research_section=WEB_RESEARCH_PROMPT_SECTION_CHAT
        if config.get("web_research_enabled")
        else "",
    )

    # Set up configuration
    run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    if config:
        run_config.update(config)

    try:
        logger.debug("Implementation agent completed successfully")
        return run_agent_with_retry(agent, prompt, run_config)
    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Implementation agent failed: %s", str(e), exc_info=True)
        raise


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
    def __enter__(self):
        _CONTEXT_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _CONTEXT_STACK.remove(self)


def check_interrupt():
    if _CONTEXT_STACK and _INTERRUPT_CONTEXT is _CONTEXT_STACK[-1]:
        raise AgentInterrupt("Interrupt requested")


def run_agent_with_retry(agent, prompt: str, config: dict) -> Optional[str]:
    """Run an agent with retry logic for API errors."""
    logger.debug("Running agent with prompt length: %d", len(prompt))
    original_handler = None
    if threading.current_thread() is threading.main_thread():
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _request_interrupt)

    max_retries = 20
    base_delay = 1

    with InterruptibleSection():
        try:
            # Track agent execution depth
            current_depth = _global_memory.get("agent_depth", 0)
            _global_memory["agent_depth"] = current_depth + 1

            for attempt in range(max_retries):
                logger.debug("Attempt %d/%d", attempt + 1, max_retries)
                check_interrupt()
                try:
                    for chunk in agent.stream(
                        {"messages": [HumanMessage(content=prompt)]}, config
                    ):
                        logger.debug("Agent output: %s", chunk)
                        check_interrupt()
                        print_agent_output(chunk)
                        if _global_memory["plan_completed"]:
                            _global_memory["plan_completed"] = False
                            _global_memory["task_completed"] = False
                            _global_memory["completion_message"] = ""
                            break
                        if _global_memory["task_completed"]:
                            _global_memory["task_completed"] = False
                            _global_memory["completion_message"] = ""
                            break
                    logger.debug("Agent run completed successfully")
                    return "Agent run completed successfully"
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
                    if attempt == max_retries - 1:
                        logger.error("Max retries reached, failing: %s", str(e))
                        raise RuntimeError(
                            f"Max retries ({max_retries}) exceeded. Last error: {e}"
                        )
                    logger.warning(
                        "API error (attempt %d/%d): %s",
                        attempt + 1,
                        max_retries,
                        str(e),
                    )
                    delay = base_delay * (2**attempt)
                    print_error(
                        f"Encountered {e.__class__.__name__}: {e}. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})"
                    )
                    start = time.monotonic()
                    while time.monotonic() - start < delay:
                        check_interrupt()
                        time.sleep(0.1)
        finally:
            # Reset depth tracking
            _global_memory["agent_depth"] = _global_memory.get("agent_depth", 1) - 1

            if (
                original_handler
                and threading.current_thread() is threading.main_thread()
            ):
                signal.signal(signal.SIGINT, original_handler)
