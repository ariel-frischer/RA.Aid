"""Prompt templates for context condensation."""

from typing import List
from langchain_core.messages import BaseMessage

# Basic condensation prompt
CONDENSATION_PROMPT = """
Summarize the following conversation chunk concisely while preserving key information, 
decisions, and context needed for future reference. Focus on facts, conclusions, and 
important details that would be needed to continue the conversation effectively.

CONVERSATION CHUNK:
{conversation_chunk}

CONCISE SUMMARY:
"""

# Advanced condensation prompt with more detailed instructions
ADVANCED_CONDENSATION_PROMPT = """
Create a concise summary of the following conversation that preserves all essential information.

Your summary should:
1. Maintain all key facts, decisions, and important context
2. Preserve any conclusions or insights reached
3. Include any specific details that would be needed to continue the conversation
4. Capture the logical flow and reasoning
5. Be significantly shorter than the original conversation

CONVERSATION CHUNK:
{conversation_chunk}

CONCISE SUMMARY:
"""

# Specialized prompt for research conversations
RESEARCH_CONDENSATION_PROMPT = """
Summarize the following research conversation concisely while preserving all critical information.

Focus on:
1. Research questions and objectives
2. Key findings and discoveries
3. Sources and references mentioned
4. Methodologies discussed
5. Conclusions and next steps

RESEARCH CONVERSATION:
{conversation_chunk}

CONCISE RESEARCH SUMMARY:
"""

# Specialized prompt for planning conversations
PLANNING_CONDENSATION_PROMPT = """
Create a concise summary of this planning conversation that preserves all essential elements.

Focus on:
1. Goals and objectives established
2. Decisions made and their rationale
3. Action items and responsibilities assigned
4. Timeline and milestones discussed
5. Constraints and requirements identified

PLANNING CONVERSATION:
{conversation_chunk}

CONCISE PLANNING SUMMARY:
"""

# Specialized prompt for implementation conversations
IMPLEMENTATION_CONDENSATION_PROMPT = """
Summarize this implementation-focused conversation concisely while preserving all technical details.

Focus on:
1. Technical requirements and specifications
2. Implementation approaches discussed
3. Code or architecture decisions
4. Technical challenges and solutions
5. Testing and validation plans

IMPLEMENTATION CONVERSATION:
{conversation_chunk}

CONCISE IMPLEMENTATION SUMMARY:
"""

def get_condensation_prompt_for_agent_type(agent_type: str) -> str:
    """Get the appropriate condensation prompt for the agent type.
    
    Args:
        agent_type: Type of agent ("default", "research", "planning", "implementation")
        
    Returns:
        str: The condensation prompt template
    """
    if agent_type == "research":
        return RESEARCH_CONDENSATION_PROMPT
    elif agent_type == "planning":
        return PLANNING_CONDENSATION_PROMPT
    elif agent_type == "implementation":
        return IMPLEMENTATION_CONDENSATION_PROMPT
    else:
        return CONDENSATION_PROMPT

def format_messages_for_condensation(messages: List[BaseMessage]) -> str:
    """Format a list of messages for condensation.
    
    Args:
        messages: List of messages to format
        
    Returns:
        str: Formatted conversation text
    """
    formatted_lines = []
    
    for msg in messages:
        role = msg.type.upper()
        content = msg.content
        
        # Handle different content types
        if isinstance(content, list):
            # Handle content that might be a list of content blocks
            content_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        content_parts.append(part.get("text", ""))
                    elif part.get("type") == "image":
                        content_parts.append("[IMAGE]")
                    elif part.get("type") == "tool_use":
                        content_parts.append(f"[TOOL USE: {part.get('name', 'unknown')}]")
                    elif part.get("type") == "tool_result":
                        content_parts.append(f"[TOOL RESULT: {part.get('content', '')}]")
                else:
                    content_parts.append(str(part))
            content = "\n".join(content_parts)
        
        formatted_lines.append(f"{role}: {content}")
    
    return "\n\n".join(formatted_lines)