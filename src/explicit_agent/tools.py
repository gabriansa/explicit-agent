import inspect
import warnings
from typing import Any, Dict, List, Type, Optional

from openai import pydantic_function_tool
from pydantic import BaseModel


class BaseTool(BaseModel):
    """
    Base class for tools.
    If no `execute` method is provided, a default implementation returning `None` will be used.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "execute" not in cls.__dict__:
            warnings.warn(
                f"Tool class '{cls.__name__}' does not implement an 'execute' method. "
                f"A default method returning None will be used."
            )

    def execute(self) -> Any:
        """Execute the tool functionality
        
        This method should be implemented by subclasses to define the tool's behavior.
        Within the implementation, you can access any defined variables as self.variable
        (e.g., self.name, self.value, etc.)

        Returns:
            `Any`: The result of the tool execution
        """
        return None


class StopTool(BaseTool):
    """
    Base class for stop tools.
    If no `execute` method is provided, a default implementation returning `None` will be used.
    """

    def execute(self) -> Any:
        """Execute the stop tool functionality

        This method should be implemented by subclasses to define the tool's behavior.
        Within the implementation, you can access any defined variables as self.variable
        (e.g., self.name, self.value, etc.)

        Returns:
            `Any`: The result of the tool execution
        """
        return None


def register_tools(tool_classes: List[Type[BaseTool]]) -> Dict[Type, Any]:
    """
    Convert tool classes to OpenAI type tools.

    Args:
        `tool_classes`: List of tool classes to register

    Returns:
        `dict` mapping tool classes to their OpenAI type tool definitions
    """
    return {cls: pydantic_function_tool(cls) for cls in tool_classes}
