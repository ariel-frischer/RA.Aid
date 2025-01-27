import logging
import os
import shlex
from dataclasses import dataclass, field
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Configuration for shell command security settings."""
    
    allowed_commands: Set[str] = field(default_factory=set)
    blocked_commands: Set[str] = field(default_factory=set)
    max_args: int = 10
    max_command_length: int = 500
    allow_env_vars: bool = False
    restricted_paths: Set[str] = field(default_factory=lambda: {"/tmp", "/var/tmp"})


class ShellSecurityValidator:
    """Validates and sanitizes shell commands for secure execution.
    
    Provides command validation, input sanitization, and basic sandboxing controls
    for shell command execution. Uses shlex for proper command parsing and implements
    configurable security policies.
    
    Attributes:
        config: SecurityConfig instance containing security settings
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize the validator with security configuration.
        
        Args:
            config: Optional SecurityConfig instance. If None, uses defaults.
        """
        self.config = config or SecurityConfig()
        
    def validate_command(self, command: str) -> bool:
        """Validate a shell command against security policies.
        
        Performs comprehensive validation including:
        - Command parsing and structure validation
        - Whitelist/blacklist checking
        - Input sanitization
        - Path and argument validation
        
        Args:
            command: The shell command string to validate
            
        Returns:
            bool: True if command passes all validation, False otherwise
            
        Raises:
            ValueError: If command is malformed or violates security policies
        """
        if not command or not isinstance(command, str):
            raise ValueError("Command must be a non-empty string")
            
        if len(command) > self.config.max_command_length:
            raise ValueError(f"Command exceeds maximum length of {self.config.max_command_length}")
            
        try:
            parsed_cmd = self.parse_command(command)
        except ValueError as e:
            logger.error(f"Command parsing failed: {e}")
            raise ValueError(f"Invalid command format: {e}")
            
        if not self.is_command_allowed(parsed_cmd[0]):
            raise ValueError(f"Command '{parsed_cmd[0]}' is not allowed")
            
        if len(parsed_cmd) > self.config.max_args + 1:
            raise ValueError(f"Too many arguments (max {self.config.max_args})")
            
        # Validate each argument
        for arg in parsed_cmd[1:]:
            self.sanitize_input(arg)
            
        return True
        
    def is_command_allowed(self, cmd: str) -> bool:
        """Check if a command is allowed based on whitelist/blacklist.
        
        Args:
            cmd: The command to check (without arguments)
            
        Returns:
            bool: True if command is allowed, False otherwise
        """
        cmd = cmd.lower()
        
        if self.config.blocked_commands and cmd in self.config.blocked_commands:
            return False
            
        if self.config.allowed_commands:
            return cmd in self.config.allowed_commands
            
        return True
        
    def sanitize_input(self, input_str: str) -> str:
        """Sanitize command input for safe execution.
        
        Performs validation and cleaning of command arguments:
        - Checks for dangerous characters
        - Validates paths
        - Sanitizes environment variables if allowed
        
        Args:
            input_str: The input string to sanitize
            
        Returns:
            str: Sanitized input string
            
        Raises:
            ValueError: If input contains dangerous content
        """
        if not input_str:
            return ""
            
        # Check for dangerous patterns
        dangerous_patterns = ["|", "&", ";", "`", "$(("]
        for pattern in dangerous_patterns:
            if pattern in input_str:
                raise ValueError(f"Input contains dangerous pattern: {pattern}")
                
        # Handle environment variables
        if "$" in input_str and not self.config.allow_env_vars:
            raise ValueError("Environment variables are not allowed")
            
        # Validate paths
        if input_str.startswith("/"):
            path = os.path.abspath(input_str)
            for restricted in self.config.restricted_paths:
                if path.startswith(restricted):
                    raise ValueError(f"Access to path {restricted} is restricted")
                    
        return input_str
        
    def parse_command(self, command: str) -> List[str]:
        """Safely parse command string using shlex.
        
        Args:
            command: The command string to parse
            
        Returns:
            List[str]: List of parsed command components
            
        Raises:
            ValueError: If command parsing fails
        """
        try:
            return shlex.split(command)
        except ValueError as e:
            raise ValueError(f"Command parsing failed: {e}")
            
    def configure(self, config: SecurityConfig) -> None:
        """Update security configuration.
        
        Args:
            config: New SecurityConfig instance
        """
        self.config = config
