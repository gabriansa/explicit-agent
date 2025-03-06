import inspect
import warnings
from typing import Any, Dict, List, Type, Optional

from openai import pydantic_function_tool
from pydantic import BaseModel


class BaseTool(BaseModel):
    """Base class for tools.

    Tools can be stateful or stateless based on their `execute` method signature:
    - If `execute` method includes a `state` parameter, it's considered stateful (e.g `def execute(state, **kwargs)`)
    - If `execute` method doesn't have a `state` parameter, it's considered stateless (e.g `def execute(**kwargs)`)
    - If no `execute` method is provided, a default implementation returning `None` will be used.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "execute" not in cls.__dict__:
            warnings.warn(
                f"Tool class '{cls.__name__}' does not implement an 'execute' method. "
                f"A default method returning None will be used."
            )

    def execute(self, state: Optional[Any] = None, **kwargs) -> Any:
        """Execute the tool functionality

        Args:
            `state`: The current state of the agent (optional, making the tool stateful)
            `**kwargs`: Tool-specific arguments defined by the tool's Pydantic fields

        Returns:
            `Any`: The result of the tool execution
        """
        pass

    @classmethod
    def is_stateful(cls) -> bool:
        """Determine if this tool is stateful by checking for a 'state' parameter in execute"""
        sig = inspect.signature(cls.execute)
        return "state" in sig.parameters


class StopTool(BaseTool):
    """Base class for stop tools.

    Tools can be stateful or stateless based on their `execute` method signature:
    - If `execute` method includes a `state` parameter, it's considered stateful (e.g `def execute(state, **kwargs)`)
    - If `execute` method doesn't have a `state` parameter, it's considered stateless (e.g `def execute(**kwargs)`)
    - If no `execute` method is provided, a default implementation returning `None` will be used.
    """

    def execute(self, state: Optional[Any] = None, **kwargs) -> Any:
        """Execute the stop tool functionality

        Args:
            `state`: The current state of the agent (optional, making the tool stateful)
            `**kwargs`: Tool-specific arguments defined by the tool's Pydantic fields

        Returns:
            `Any`: The result of the tool execution
        """
        pass


def register_tools(tool_classes: List[Type[BaseTool]]) -> Dict[Type, Any]:
    """
    Convert tool classes to OpenAI type tools.

    Args:
        `tool_classes`: List of tool classes to register

    Returns:
        `dict` mapping tool classes to their OpenAI type tool definitions
    """
    return {cls: pydantic_function_tool(cls) for cls in tool_classes}
