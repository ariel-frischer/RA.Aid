"""AgentRunner class for encapsulated agent execution with retries, interruptions, and test handling."""

import signal
import threading
from typing import Any, Dict, Optional, Union

from langchain_core.messages import HumanMessage
from rich.console import Console

from ra_aid.agents.output_summarizer import OutputSummarizer

from ra_aid.console.output import print_agent_output
from ra_aid.exceptions import AgentInterrupt
from ra_aid.interruptible_section import InterruptibleSection, _request_interrupt
from ra_aid.logging_config import get_logger
from ra_aid.retry_manager import RetryManager
from ra_aid.test_executor import TestExecutor

logger = get_logger(__name__)
console = Console()


class AgentRunner:
    """Encapsulates agent execution flow with retries, interruptions, and test integration.

    Responsibilities:
    - Manages agent execution lifecycle
    - Handles interrupt routing and cleanup
    - Coordinates retries and test execution
    - Maintains execution state
    - Processes and summarizes shell command outputs
    
    The AgentRunner integrates with OutputSummarizer to provide enhanced shell command
    output processing capabilities. This includes:
    - Automatic output pattern recognition
    - Suggestion generation for common issues
    - Caching of successful output patterns
    - Redis-based distributed caching support
    
    Example Usage:
        # Configure with Redis-enabled summarizer
        config = {
            'summarizer': {
                'model': 'claude-3-haiku-20240307',
                'use_redis': True,
                'redis_config': {
                    'host': 'localhost',
                    'port': 6379
                }
            }
        }
        
        runner = AgentRunner(
            agent=my_agent,
            initial_prompt="Execute task",
            config=config
        )
        
        # Run agent with output processing
        result = runner.run()
        
    Best Practices:
    1. Configure appropriate retry limits and backoff
    2. Enable Redis caching for production deployments
    3. Monitor and log output patterns
    4. Handle interrupts gracefully
    5. Implement proper cleanup on completion
    """

    def __init__(
        self,
        agent: Any,
        initial_prompt: str,
        config: Dict[str, Any],
        retry_manager: Optional[RetryManager] = None,
        test_executor: Optional[TestExecutor] = None,
    ):
        self.agent = agent
        self.original_prompt = initial_prompt
        self.current_prompt = initial_prompt
        self.config = config
        self.test_attempts = 0
        self.agent_depth = 0
        self._original_signal_handler = None
        self._interrupt_section = InterruptibleSection()

        # Strategy components
        self.retry_manager = retry_manager or RetryManager()
        self.test_executor = test_executor or TestExecutor(config)
        
        # Output processing
        summarizer_config = config.get('summarizer', {})
        self.output_summarizer = OutputSummarizer(
            model=summarizer_config.get('model', 'claude-3-haiku-20240307'),
            use_redis=summarizer_config.get('use_redis', False),
            redis_config=summarizer_config.get('redis_config')
        )

    def run(self) -> Optional[str]:
        """Execute the agent with full lifecycle management."""
        logger.debug("Running agent with prompt length: %d", len(self.current_prompt))

        self._setup_signal_handling()

        try:
            with self._interrupt_section:
                return self._run_agent_loop()
        finally:
            self._cleanup_resources()

    def _run_agent_loop(self) -> Optional[str]:
        """Main execution loop handling retries and test integration."""
        while True:
            should_break = self._execute_agent_iteration()
            print(f"should_break={should_break}")

            if should_break:
                logger.debug("Agent run completed successfully")
                return "Agent run completed successfully"

            execute_result = self.test_executor.execute(
                self.test_attempts, self.original_prompt
            )

            continue_execution, prompt, test_attempts = execute_result

            print(f"continue_execution={continue_execution}")

            self._update_from_test_state(continue_execution, prompt, test_attempts)

            if should_break or not continue_execution:
                logger.debug("Agent run completed successfully")
                return "Agent run completed successfully"

    def _execute_agent_iteration(self) -> bool:
        """Execute one agent iteration with retry handling."""
        return self.retry_manager.execute(
            self._run_agent_iteration,
            self.agent,
            self.current_prompt,
            self.config,
            self._interrupt_section,
        )

    def _run_agent_iteration(self, agent, prompt, config, section) -> bool:
        """Single agent iteration with interrupt checks."""
        for chunk in agent.stream({"messages": [HumanMessage(content=prompt)]}, config):
            self._check_interrupts(section)
            self._process_chunk(chunk)

            if self._check_completion_state():
                return True
        return False

    def _setup_signal_handling(self):
        """Configure system signal handling for main thread."""
        if threading.current_thread() is threading.main_thread():
            self._original_signal_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _request_interrupt)

    def _cleanup_resources(self):
        """Clean up system resources and global state."""
        if (
            self._original_signal_handler
            and threading.current_thread() is threading.main_thread()
        ):
            signal.signal(signal.SIGINT, self._original_signal_handler)

    def _update_from_test_state(
        self, continue_execution: bool, prompt: str, test_attempts: int
    ):
        """Update runner state from test execution results."""
        self.test_attempts = test_attempts
        self.current_prompt = prompt

    def _check_interrupts(self, section: InterruptibleSection):
        """Check if execution has been interrupted."""
        if section.is_interrupted():
            raise AgentInterrupt("Agent execution interrupted")

    def _process_chunk(self, chunk: Dict[str, Any]):
        """Process a single chunk of agent output.
        
        Handles different types of chunks including shell command outputs.
        Shell outputs are processed through OutputSummarizer for additional insights.
        
        Args:
            chunk: Dictionary containing output chunk data
            
        Examples:
            # Process shell command output
            >>> chunk = {
            ...     'tools': {
            ...         'messages': [{
            ...             'content': 'ls: cannot access file.txt'
            ...         }]
            ...     }
            ... }
            >>> runner._process_chunk(chunk)
            # Generates suggestions and caches pattern
            
            # Process regular agent output
            >>> chunk = {
            ...     'agent': {
            ...         'messages': ['Task completed']
            ...     }
            ... }
            >>> runner._process_chunk(chunk)
            # Prints output without summarization
            
            # Handle empty or invalid chunks
            >>> runner._process_chunk({})
            # Safely handles missing data
        """
        logger.debug("Agent output: %s", chunk)
        
        # Process shell command outputs
        if isinstance(chunk.get('tools'), dict):
            tools_data = chunk['tools']
            if isinstance(tools_data.get('messages'), list):
                for msg in tools_data['messages']:
                    if hasattr(msg, 'content') and msg.content:
                        try:
                            # Process shell output through summarizer
                            summary = self.output_summarizer._generate_suggestions(msg.content)
                            if summary:
                                logger.debug("Output summary: %s", summary)
                                # Cache successful patterns if needed
                                self._cache_output_pattern(msg.content, summary)
        
        # Standard output processing
        print_agent_output(chunk)
        
    def _cache_output_pattern(self, output: str, summary: Union[str, List[str]]):
        """Cache successful output patterns for future reference."""
        try:
            if isinstance(summary, list) and summary:
                self.output_summarizer._cache_set(
                    self.output_summarizer._get_cache_key(output),
                    summary
                )
        except Exception as e:
            logger.debug(f"Failed to cache output pattern: {e}")

    def _check_completion_state(self) -> bool:
        """Check if any completion states are set."""
        from ra_aid.tools.memory import _global_memory

        if _global_memory.get("plan_completed"):
            _global_memory.update(
                {
                    "plan_completed": False,
                    "task_completed": False,
                    "completion_message": "",
                }
            )
            return True

        if _global_memory.get("task_completed"):
            _global_memory.update({"task_completed": False, "completion_message": ""})
            return True

        return False
