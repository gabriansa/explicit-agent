import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Union

from openai import pydantic_function_tool
from pydantic import BaseModel


class StatelessTool(BaseModel, ABC):
    """Base class for tools that combines model definition and implementation"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "execute" not in cls.__dict__:
            raise TypeError(
                f"Tool class '{cls.__name__}' must implement an 'execute' method. Tools require an execute method to define their functionality."
            )

    @staticmethod
    @abstractmethod
    def execute(**kwargs) -> Any:
        """Execute the tool functionality

        Args:
            `**kwargs`: Tool-specific arguments

        Returns:
            `Any`: The result of the tool execution
        """
        pass


class StatefulTool(BaseModel, ABC):
    """Base class for tools that need access to the agent's state"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "execute" not in cls.__dict__:
            raise TypeError(
                f"Tool class '{cls.__name__}' must implement an 'execute' method. Tools require an execute method to define their functionality."
            )

        sig = inspect.signature(cls.execute)
        params = list(sig.parameters.keys())
        if "state" not in params:
            raise TypeError(
                f"Execute method in {cls.__name__} must have 'state' parameter"
            )

    @staticmethod
    @abstractmethod
    def execute(state: Any, **kwargs) -> Any:
        """Execute the tool functionality with access to agent state

        Args:
            `state`: The current state of the agent
            `**kwargs`: Tool-specific arguments

        Returns:
            `Any`: The result of the tool execution
        """
        pass


class StopStatelessTool(StatelessTool):
    """Base class for tools that signals to stop agent execution when called"""

    @staticmethod
    @abstractmethod
    def execute(**kwargs) -> Any:
        """Execute the stop tool functionality

        Args:
            `**kwargs`: Tool-specific arguments

        Returns:
            `Any`: The result of the tool execution
        """
        pass


class StopStatefulTool(StatefulTool):
    """Base class for tools that signals to stop agent execution when called"""

    @staticmethod
    @abstractmethod
    def execute(state: Any, **kwargs) -> Any:
        """Execute the stop tool functionality with access to agent state

        Args:
            `state`: The current state of the agent
            `**kwargs`: Tool-specific arguments

        Returns:
            `Any`: The result of the tool execution
        """
        pass


def register_tools(
    tool_classes: List[
        Type[Union[StatelessTool, StatefulTool, StopStatelessTool, StopStatefulTool]]
    ],
) -> Dict[Type, Any]:
    """
    Convert tool classes to OpenAI type tools.

    Args:
        `tool_classes`: List of tool classes to register

    Returns:
        `dict` mapping tool classes to their OpenAI type tool definitions
    """
    return {cls: pydantic_function_tool(cls) for cls in tool_classes}
