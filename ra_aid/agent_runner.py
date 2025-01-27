"""AgentRunner class for encapsulated agent execution with retries, interruptions, and test handling."""

import signal
import threading
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from rich.console import Console

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
        self.agent_depth = 0
        self._original_signal_handler = None
        self._interrupt_section = InterruptibleSection()

        # Strategy components
        self.retry_manager = retry_manager or RetryManager()
        self.test_executor = test_executor or TestExecutor(config)

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
            agent_complete = self._execute_agent_iteration()
            if agent_complete:
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
        self, should_break: bool, prompt: str
    ):
        """Update runner state from test execution results."""
        self.current_prompt = prompt

    def _check_interrupts(self, section: InterruptibleSection):
        """Check if execution has been interrupted."""
        if section.is_interrupted():
            raise AgentInterrupt("Agent execution interrupted")

    def _process_chunk(self, chunk: Dict[str, Any]):
        """Process a single chunk of agent output."""
        logger.debug("Agent output: %s", chunk)
        print_agent_output(chunk)

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
