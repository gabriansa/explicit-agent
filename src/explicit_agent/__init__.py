"""
Explicit Agent Framework - A framework for creating explicit tool-using agents

This framework provides a clean, efficient way to build tool-using agents with
explicit control over tool execution, state management, and agent flow.
"""

import logging
import importlib.metadata

logging.getLogger("explicit_agent").addHandler(logging.NullHandler())

from .agent import ExplicitAgent
from .tools import (
    BaseTool,
    StopTool,
    register_tools,
)

# Get version from pyproject.toml
__version__ = importlib.metadata.version("explicit-agent")

__all__ = [
    "ExplicitAgent",
    "BaseTool",
    "StopTool",
    "register_tools",
]
