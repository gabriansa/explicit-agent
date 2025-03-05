"""
Explicit Agent Framework - A framework for creating explicit tool-using agents

This framework provides a clean, efficient way to build tool-using agents with
explicit control over tool execution, state management, and agent flow.
"""

import logging

# Configure basic logging
logging.getLogger('explicit_agent').addHandler(logging.NullHandler())

from .agent import ExplicitAgent
from .tools import BaseTool, StateAwareTool, StopTool, register_tools

# Define the version as a tuple for easy comparison and a string for display
__version_info__ = (0, 1, 0)
__version__ = '.'.join(str(c) for c in __version_info__)

__all__ = [
    "ExplicitAgent",
    "BaseTool",
    "StateAwareTool", 
    "StopTool",
    "register_tools",
]
