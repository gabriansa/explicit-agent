# Hi, this is Explicit Agent

A minimalist, transparent framework for building AI agents with full user control and zero abstraction layers - yes ZERO!

![Explicit Agent](assets/explicit.png)

## Table of Contents
- [Why Explicit Agent?](#why-explicit-agent)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [How to use it](#how-to-use-it)
- [Core Concepts](#core-concepts)
  - [State Management](#state-management)
  - [Tool Types](#tool-types)
  - [Tool Return Values](#tool-return-values)
- [When is the Explicit Agent Framework Useful?](#when-is-the-explicit-agent-useful)


## Why Explicit Agent?

Most agentic frameworks are overengineered with layers of abstraction that obscure what's actually happening. Explicit Agent cuts through the BS to provide:

- **Complete transparency**: No hidden prompts or "magic" under the hood
- **Full control**: You define exactly how your agent behaves
- **Minimal infrastructure**: Only the essentials needed to run capable AI agents
- **Simplicity first**: Ability to build complex behaviors from simple, understandable components

This framework provides the minimum viable infrastructure for running AI agents while maintaining full visibility into their operation.

At the end of the day, an agent should be able to solve a task autonomously given a set of tools. And this is it.

## Get Started

### Installation

You can install Explicit Agent directly from [PyPI](https://pypi.org/project/explicit-agent/):

```bash
pip install explicit-agent
```

Or install from source (reccomended):

```bash
# Clone the repository
git clone https://github.com/gabriansa/explicit-agent.git
cd explicit-agent

# Install the package
pip install -e .
```

### How to use it

```python
from explicit_agent import ExplicitAgent
from explicit_agent.tools import BaseTool, StopTool

from pydantic import Field


# ========= DEFINING STATE =========
# Define a simple state object to be shared across tools
state = {"result": None}

# ========= DEFINING TOOLS =========
# Tools are the actions your agent can perform
# Each tool is a Pydantic model with an execute method

# BaseTool - Standard tool that performs an action but doesn't stop the agent
class Multiply(BaseTool):
    """Multiply two numbers"""
    # Define the parameters this tool accepts - these become required fields
    a: int | float = Field(..., description="The first number to multiply")
    b: int | float = Field(..., description="The second number to multiply")

    # The execute method defines what happens when this tool is called
    def execute(self):
        # Calculate the result
        result = self.a * self.b
        # Save result to the state so other tools can access it later
        state["result"] = result
        # Return value is what gets sent back to the LLM
        return f"Multiplied {self.a} × {self.b} = {result}"

# StopTool - Special tool type that signals the agent to stop execution
# Use this for final actions or to return results to the user
class ShowResult(StopTool):
    """Show the final result"""

    # This tool doesn't need parameters because it gets data from state
    def execute(self):
        # Return the final result that was stored in state by previous tool calls
        return f"Final result: {state['result']}"


# ========= SYSTEM PROMPT =========
# The system prompt defines the agent's personality and instructions
# This is the first message sent to the LLM - be explicit about available tools
system_prompt = """
You are a calculator.
These are the tools you can use:
- Multiply
- ShowResult

When you are done with the calculation, use the `ShowResult` tool to show the final result.
"""

# ========= AGENT INITIALIZATION =========
# Initialize the agent with key parameters
agent = ExplicitAgent(
    api_key=api_key,  # Your API key for the LLM provider
    base_url=base_url,  # Base URL for the provider (e.g., OpenAI, Azure, etc.)
    verbose=True # Print logs of what is happening
)

# ========= USER PROMPT =========
# This is the task you want the agent to perform
prompt = """
Do the following calculations:
1. Multiply 3294 by 1023
2. Multiply the result by 29218
3. Show the final result
"""

# ========= AGENT EXECUTION =========
# Run the agent with the prompt and tools
agent.run(
    model="openai/gpt-4o-mini",  # LLM model to use
    prompt=prompt,               # User's instructions
    system_prompt=system_prompt,  # Instructions for the agent
    budget=10,                   # Maximum number of steps (tool calls) before forced termination
    tools=[Multiply, ShowResult], # List of available tools
)
# When execution completes, the state variable contains any information preserved across tool calls
# A StopTool will trigger completion, or the agent will stop when budget is exhausted
```

## Core Concepts
Explicit Agent is built around a few simple concepts.

![Explicit Agent Framework](assets/framework.png)

### State Management

In Explicit Agent, state management is completely up to you. The framework doesn't impose any specific state structure, giving you full control over how information is shared between tools:

- **User-Defined State**: You define your own state object (typically a dictionary) to persist data across tool calls.
- **Flexible Implementation**: You can use global variables, class attributes, or any other approach that suits your application.
- **Direct Manipulation**: Tools can directly access and modify this state, creating a simple and transparent flow of information.

This approach offers several benefits:
- **Clarity**: The state and its transformations are explicitly visible in your code
- **Control**: You decide exactly what data is stored and how it's structured
- **Simplicity**: No hidden state management mechanisms to debug or understand

Example state implementations:
```python
# Simple global state
state = {"counter": 0, "results": []}

# State as part of a class
class MyAgent:
    def __init__(self):
        self.state = {"counter": 0, "results": []}
        self.agent = ExplicitAgent(...)
    
    def run_task(self, prompt):
        return self.agent.run(
            prompt=prompt,
            tools=[...],  # Tools that can access self.state
        )
```

### Tool Types

- **`BaseTool`**: This is the base class for creating tools.
- **`StopTool`**: This is the base class for creating stop tools. Stop tools are extremely important because they are the ones that signal when the agent should stop execution.

### Tool Return Values

A crucial aspect of the Explicit Agent framework is how tool return values are handled:

- **LLM Feedback Loop**: Whatever a tool returns (the output of its `execute` method) is sent back to the LLM as part of its conversation history.
- **Decision Making**: The LLM uses these return values to inform its next actions, creating a dynamic feedback loop.
- **Rich Responses**: Tools can return strings, structured data, or any format that would help the LLM understand the result of the action.

This means tools serve two purposes:
1. They perform actions that affect the external world or update the shared state
2. They provide information back to the LLM to guide its decision-making process

### Execution Flow

1. The agent receives a prompt from the user
2. The agent generates tool calls based on the prompt and system instructions
3. The tools are executed, potentially updating the user-defined state
4. The results from the tools are fed back to the agent, informing subsequent decisions
5. This continues until a `StopTool` is called or the budget is exhausted

## Examples

For more advanced usage and detailed documentation, see the [examples](examples) directory:

- [Calculator Example](examples/calculator.py): A simple calculator agent that performs arithmetic operations
- [Shopping Cart Example](examples/shopping_cart.py): A more complex example of a shopping assistant that manages a cart
- [PhD Assistant](examples/phd_assistant.py): An advanced example of an agent that helps with academic research and writing

# When is the Explicit Agent Useful?

The Explicit Agent shines in situations where tasks mimic human workflows that involve exploration, decision-making, and adaptability.
- **Unstructured or Semi-Structured Tasks**: The framework excels when tasks don't follow a rigid, predefined path and require the agent to adjust its approach based on new information or evolving requirements.

- **Multi-Step Processes with Tool Usage**: It's ideal for tasks that involve multiple actions or tools, where the agent must decide which tool to use and in what order, depending on the current context.

- **State Management Across Steps**: The agent's ability to work with user-defined state allows for tracking progress, sharing information between tools, and building on previous results.

- **Human-Like Reasoning**: It's best suited for scenarios where judgment, exploration, or iterative refinement—qualities typically associated with human decision-making—are beneficial.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
