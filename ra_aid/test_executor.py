"""TestExecutor module for handling test command execution and retry logic."""

from dataclasses import dataclass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.logging_config import get_logger

from .config import DEFAULT_MAX_TEST_CMD_RETRIES
from .tools.human import ask_human
from .tools.shell import run_shell_command

logger = get_logger(__name__)
console = Console()


@dataclass
class TestState:
    """State for test execution."""

    prompt: str
    test_attempts: int
    auto_test: bool
    should_break: bool = False


class TestExecutor:
    """Handles test command execution and retry logic."""

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.config = config
        self.test_cmd = config.get("test_cmd")
        self.max_test_retries = config.get(
            "max_test_cmd_retries", DEFAULT_MAX_TEST_CMD_RETRIES
        )
        self.auto_test = config.get("auto_test", False)
        print(f"test_cmd={self.test_cmd}")
        print(f"auto_test={self.auto_test}")

    def display_test_failure(self, attempts: int) -> None:
        """Display test failure message.

        Args:
            attempts: Current number of attempts
        """
        console.print(
            Panel(
                Markdown(
                    f"Test failed. Attempt number {attempts} of {self.max_test_retries}. Retrying and informing of failure output"
                ),
                title="🔎 User Defined Test",
                border_style="red bold",
            )
        )

    def handle_test_failure(self, state: TestState, test_result: dict) -> TestState:
        """Handle test command failure.

        Args:
            state: Current test state
            test_result: Test command result

        Returns:
            Updated test state
        """
        state.prompt = f"{state.prompt}. Previous attempt failed with: <test_cmd_stdout>{test_result['output']}</test_cmd_stdout>"
        self.display_test_failure(state.test_attempts)
        state.should_break = False
        return state

    def run_test_command(self, state: TestState) -> TestState:
        """Run test command and handle result.

        Args:
            state: Current test state

        Returns:
            Updated test state
        """
        try:
            test_result = run_shell_command(self.test_cmd)
            state.test_attempts += 1

            if not test_result["success"]:
                return self.handle_test_failure(state, test_result)

            state.should_break = True
            return state

        except Exception as e:
            logger.warning(f"Test command execution failed: {str(e)}")
            state.test_attempts += 1
            state.should_break = True
            return state

    def handle_user_response(self, response: str, state: TestState) -> TestState:
        """Handle user's response to test prompt.

        Args:
            response: User's response (y/n/a)
            state: Current test state

        Returns:
            Updated test state
        """
        response = response.strip().lower()

        if response == "n":
            state.should_break = True
            return state

        if response == "a":
            state.auto_test = True
            return self.run_test_command(state)

        if response == "y":
            return self.run_test_command(state)

        return state

    def check_max_retries(self, attempts: int) -> bool:
        """Check if max retries reached.

        Args:
            attempts: Current number of attempts

        Returns:
            True if max retries reached
        """
        if attempts >= self.max_test_retries:
            logger.warning("Max test retries reached")
            return True
        return False

    def execute(
        self, test_attempts: int, original_prompt: str
    ) -> tuple[bool, str, bool, int]:
        """Execute test command if configured.

        Returns:
            Tuple containing:
            - bool: Whether to continue (True) or retry (False)
            - str: The prompt to use
            - bool: Updated auto_test flag
            - int: Updated test attempts count
        """
        print("RUNNING test executor!")
        state = TestState(
            prompt=original_prompt,
            test_attempts=test_attempts,
            auto_test=self.auto_test,
        )

        if not self.test_cmd:
            state.should_break = True
            print(f"should_break={self.should_break}")
            return (
                state.should_break,
                state.prompt,
                state.auto_test,
                state.test_attempts,
            )

        if not self.auto_test:
            print()
            response = ask_human(
                "Would you like to run the test command? (y=yes, n=no, a=enable auto-test)"
            )
            state = self.handle_user_response(response, state)
        else:
            if self.check_max_retries(test_attempts):
                print("BREAKING EARLY")
                state.should_break = True
            else:
                state = self.run_test_command(state)

        self.auto_test = state.auto_test
        return state.should_break, state.prompt, self.auto_test, state.test_attempts
