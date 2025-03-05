"""
Explicit Agent Framework - A framework for creating explicit tool-using agents
"""

from .agent import ExplicitAgent
from .tools import BaseTool, StateAwareTool, StopTool, register_tools

__all__ = [
    "ExplicitAgent",
    "BaseTool",
    "StateAwareTool", 
    "StopTool",
    "register_tools",
]

__version__ = "0.1.0"
