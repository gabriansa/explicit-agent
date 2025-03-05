from typing import List, Type, Any
from openai import pydantic_function_tool
from pydantic import BaseModel


class BaseTool(BaseModel):
    """Base class for all tools that combines model definition and implementation"""
    
    @classmethod
    def execute(cls, **kwargs) -> Any:
        """
        Implement this method in subclasses to define the tool's behavior.
        
        Args:
            `**kwargs`: Tool-specific arguments
            
        Returns:
            `Any`: The result of the tool execution
        """
        raise NotImplementedError("Tool must implement execute method")


class StateAwareTool(BaseTool):
    """Base class for tools that need access to the agent's state"""
    
    @classmethod
    def execute(cls, state: dict, **kwargs) -> dict:
        """
        Execute the tool with access to the agent's state.
        Must return the updated state as a dictionary.
        
        Args:
            `state`: The current state dictionary
            `**kwargs`: Tool-specific arguments
            
        Returns:
            `dict`: The updated state
        """
        raise NotImplementedError("Tool must implement execute method")


class StopTool(BaseModel):
    """Tool that signals to stop agent execution when called"""


def register_tools(tool_classes: List[Type[StopTool | BaseTool | StateAwareTool]]) -> dict:
    """
    Convert tool classes to OpenAI tools.
    
    Args:
        `tool_classes`: List of tool classes to register
        
    Returns:
        `dict` mapping tool classes to their OpenAI tool definitions
    """
    return {cls: pydantic_function_tool(cls) for cls in tool_classes}
