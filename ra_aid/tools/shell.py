import re
from typing import Dict, Union, Optional, Tuple
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from ra_aid.tools.memory import _global_memory
from ra_aid.proc.interactive import run_interactive_command
from ra_aid.text.processing import truncate_output
from ra_aid.console.cowboy_messages import get_cowboy_message

console = Console()

def is_test_command(command: str) -> bool:
    """Detect if a command is a test execution command.
    
    Args:
        command: The shell command string
        
    Returns:
        bool: True if command appears to be a test command
    """
    test_patterns = [
        r'^pytest\b',
        r'^python\s+-m\s+pytest\b', 
        r'^python\s+-m\s+unittest\b',
        r'^nosetests\b',
        r'npm\s+test\b',
        r'yarn\s+test\b',
        r'go\s+test\b',
        r'cargo\s+test\b'
    ]
    
    return any(re.search(pattern, command.strip()) for pattern in test_patterns)

def parse_test_output(output: str) -> Dict[str, Union[int, str]]:
    """Parse test command output to extract key metrics.
    
    Args:
        output: Raw test command output string
        
    Returns:
        Dict containing parsed metrics:
        - total: Total number of tests
        - passed: Number of passed tests
        - failed: Number of failed tests
        - errors: Error details if any
        - summary: One-line summary
    """
    # Initialize defaults
    metrics = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors": "",
        "summary": "No test results found"
    }
    
    # Look for pytest-style output
    pytest_pattern = r'=+\s*(\d+)\s+passed,?\s*(\d+)\s+failed,?\s*(\d+)\s+error'
    if match := re.search(pytest_pattern, output, re.MULTILINE):
        passed, failed, errors = map(int, match.groups())
        metrics.update({
            "total": passed + failed + errors,
            "passed": passed,
            "failed": failed + errors,
            "summary": f"{passed} passed, {failed} failed, {errors} errors"
        })
        
        # Extract error details if present
        if failed or errors:
            error_sections = re.findall(r'_{10,}\n(.*?)\n={10,}', output, re.DOTALL)
            metrics["errors"] = "\n".join(error_sections)
    
    return metrics

def format_test_results(metrics: Dict[str, Union[int, str]], output: str) -> Tuple[str, Panel]:
    """Format test results for display.
    
    Args:
        metrics: Parsed test metrics
        output: Raw command output
        
    Returns:
        Tuple of (formatted_output, summary_panel)
    """
    # Create summary panel
    status = "✅ PASS" if metrics["failed"] == 0 else "❌ FAIL"
    summary = Panel(
        f"{status}\n\n"
        f"Total Tests: {metrics['total']}\n"
        f"Passed: {metrics['passed']}\n"
        f"Failed: {metrics['failed']}\n\n"
        f"{metrics['summary']}",
        title="🧪 Test Results",
        border_style="green" if metrics["failed"] == 0 else "red"
    )
    
    # Format detailed output
    if metrics["failed"] > 0 and metrics["errors"]:
        formatted_output = metrics["errors"]
    else:
        formatted_output = truncate_output(output)
        
    return formatted_output, summary

@tool
def run_shell_command(command: str) -> Dict[str, Union[str, int, bool]]:
    """Execute a shell command and return its output.

    Important notes:
    1. Try to constrain/limit the output. Output processing is expensive, and infinite/looping output will cause us to fail.
    2. When using commands like 'find', 'grep', or similar recursive search tools, always exclude common 
       development directories and files that can cause excessive output or slow performance:
       - Version control: .git
       - Dependencies: node_modules, vendor, .venv
       - Cache: __pycache__, .cache
       - Build: dist, build
       - Environment: .env, venv, env
       - IDE: .idea, .vscode
    3. Avoid doing recursive lists, finds, etc. that could be slow and have a ton of output. Likewise, avoid flags like '-l' that needlessly increase the output. But if you really need to, you can.
    4. Add flags e.g. git --no-pager in order to reduce interaction required by the human.
    """
    is_test = is_test_command(command)
    # Check if we need approval
    cowboy_mode = _global_memory.get('config', {}).get('cowboy_mode', False)
    
    if cowboy_mode:
        console.print("")
        console.print(" " + get_cowboy_message())
        console.print("")

    # Show just the command in a simple panel
    console.print(Panel(command, title="🐚 Shell", border_style="bright_yellow"))
    
    if not cowboy_mode:
        choices = ["y", "n", "c"]
        response = Prompt.ask(
            "Execute this command? (y=yes, n=no, c=enable cowboy mode for session)",
            choices=choices,
            default="y",
            show_choices=True,
            show_default=True
        )
        
        if response == "n":
            print()
            return {
                "output": "Command execution cancelled by user",
                "return_code": 1,
                "success": False
            }
        elif response == "c":
            _global_memory['config']['cowboy_mode'] = True
            console.print("")
            console.print(" " + get_cowboy_message())
            console.print("")
    
    try:
        print()
        output, return_code = run_interactive_command(['/bin/bash', '-c', command])
        decoded_output = output.decode() if output else ""
        print()
        
        if is_test:
            # Parse and format test results
            metrics = parse_test_output(decoded_output)
            formatted_output, summary_panel = format_test_results(metrics, decoded_output)
            
            # Display summary panel
            console.print(summary_panel)
            print()
            
            return {
                "output": formatted_output,
                "return_code": return_code,
                "success": return_code == 0 and metrics["failed"] == 0
            }
        else:
            return {
                "output": truncate_output(decoded_output),
                "return_code": return_code,
                "success": return_code == 0
            }
    except Exception as e:
        print()
        console.print(Panel(str(e), title="❌ Error", border_style="red"))
        return {
            "output": str(e),
            "return_code": 1,
            "success": False
        }
