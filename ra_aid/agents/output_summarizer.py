"""Output summarizer for shell command outputs using Claude 3 Haiku.

This module provides functionality to process, summarize and cache shell command outputs
using Claude 3 Haiku. It supports both in-memory and Redis-based caching strategies.

Example Usage:

    # Basic usage with in-memory cache
    summarizer = OutputSummarizer()
    output = "ls: cannot access '/nonexistent': No such file or directory"
    suggestions = summarizer._generate_suggestions(output)
    
    # Setup with Redis caching
    redis_config = {
        "host": "localhost",
        "port": 6379,
        "db": 0
    }
    summarizer = OutputSummarizer(
        model="claude-3-haiku-20240307",
        use_redis=True,
        redis_config=redis_config
    )
    
    # Process output and cache patterns
    result = summarizer.process_output(output)
    cached = summarizer._cache_get(summarizer._get_cache_key(output))

Cache Management Best Practices:
1. Use Redis for production environments with multiple processes
2. Implement cache eviction policies based on access patterns
3. Monitor cache hit rates and adjust caching strategy
4. Consider cache warming for common patterns
5. Implement cache size limits to prevent memory issues
6. Use consistent hashing for cache keys
7. Handle cache failures gracefully
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Union

from anthropic import Anthropic
from redis import Redis

from ra_aid.logging_config import get_logger

logger = get_logger(__name__)


class OutputSummarizer:
    """Summarizes and processes shell command outputs using Claude 3 Haiku.
    
    Provides functionality to:
    - Process and summarize shell command output
    - Generate cleanup suggestions
    - Cache results in memory or Redis
    - Handle common error cases
    
    Attributes:
        model: The Claude model to use
        cache: In-memory cache dictionary
        redis_client: Optional Redis client
        anthropic: Anthropic client for API calls
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        use_redis: bool = False,
        redis_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the summarizer with model and caching config.
        
        Args:
            model: Claude model identifier to use
            use_redis: Whether to use Redis for caching
            redis_config: Redis configuration dictionary
        """
        self.model = model
        self.cache: Dict[str, Any] = {}
        self.redis_client: Optional[Redis] = None
        
        if use_redis:
            if not redis_config:
                redis_config = {"host": "localhost", "port": 6379, "db": 0}
            try:
                self.redis_client = Redis(**redis_config)
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                logger.warning("Falling back to in-memory cache")
                self.redis_client = None

        try:
            self.anthropic = Anthropic()
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise

    def process_output(self, output: str) -> Dict[str, Any]:
        """Process shell command output and return summary with suggestions.
        
        Args:
            output: Raw shell command output string
            
        Returns:
            Dictionary containing:
            - summary: Brief summary of the output
            - suggestions: List of cleanup suggestions
            - error: Optional error information
            
        Raises:
            ValueError: If output is empty or invalid
        """
        if not output or not isinstance(output, str):
            raise ValueError("Invalid output provided")

        cache_key = self._get_cache_key(output)
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        try:
            # Generate summary using Claude
            message = self.anthropic.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": f"Summarize this shell command output briefly and identify key points:\n\n{output}"
                }]
            )
            summary = message.content[0].text

            # Generate cleanup suggestions
            suggestions = self._generate_suggestions(output)

            result = {
                "summary": summary,
                "suggestions": suggestions,
                "error": None
            }

            self._cache_set(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error processing output: {e}")
            return {
                "summary": "",
                "suggestions": [],
                "error": str(e)
            }

    def _generate_suggestions(self, output: str) -> List[str]:
        """Generate cleanup/filtering suggestions for the output.
        
        Analyzes shell command output and generates suggestions for improving
        readability and usability. Uses Claude 3 Haiku for analysis.
        
        Args:
            output: Raw command output to analyze
            
        Returns:
            List of suggestion strings
            
        Examples:
            >>> summarizer = OutputSummarizer()
            >>> output = "ls: cannot access '/nonexistent': No such file or directory"
            >>> suggestions = summarizer._generate_suggestions(output)
            >>> print(suggestions)
            ['Check if directory exists before accessing',
             'Use -p flag with mkdir to create parent directories',
             'Consider using path.exists() for validation']
            
            # Handle empty output
            >>> suggestions = summarizer._generate_suggestions("")
            >>> print(suggestions)
            []
            
            # Process complex multi-line output
            >>> output = '''
            ... make: Entering directory '/project'
            ... gcc -c main.c
            ... main.c:10: warning: unused variable 'x'
            ... make: Leaving directory '/project'
            ... '''
            >>> suggestions = summarizer._generate_suggestions(output)
            >>> print(suggestions)
            ['Address compiler warning about unused variable',
             'Consider using -Wall for additional warnings']
        """
        try:
            message = self.anthropic.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": f"Suggest ways to clean up or filter this command output to make it more readable:\n\n{output}"
                }]
            )
            
            # Split suggestions into list
            suggestions = [
                s.strip() for s in message.content[0].text.split("\n")
                if s.strip() and not s.strip().startswith(("-", "*", "•"))
            ]
            return suggestions[:5]  # Limit to top 5 suggestions

        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []

    def _get_cache_key(self, output: str) -> str:
        """Generate cache key from output string.
        
        Args:
            output: String to generate key from
            
        Returns:
            Cache key string
        """
        return hashlib.md5(output.encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[Any]:
        """Get item from cache (Redis or in-memory).
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value or None if not found
        """
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                return eval(value.decode()) if value else None
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                return self.cache.get(key)
        return self.cache.get(key)

    def _cache_set(self, key: str, value: Any) -> None:
        """Store item in cache (Redis or in-memory).
        
        Args:
            key: Cache key
            value: Value to store
        """
        if self.redis_client:
            try:
                self.redis_client.set(key, str(value))
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        self.cache[key] = value
