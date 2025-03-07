# Hi, this is Explicit Agent

A minimalist, transparent framework for building AI agents with full user control and zero abstraction layers - yes ZERO!

![Explicit Agent](assets/explicit.png)

## Table of Contents
- [Why Explicit Agent?](#why-explicit-agent)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [How to use it](#how-to-use-it)
- [Core Concepts](#core-concepts)
  - [Agent State](#agent-state)
  - [Tool Types](#tool-types)
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
    # If the method has a 'state' parameter, it's stateful and can modify agent state
    def execute(self, state):
        # Save result to the agent's state so other tools can access it later
        state["result"] = self.a * self.b
        # Return value is what gets sent back to the LLM
        return self.a * self.b

# StopTool - Special tool type that signals the agent to stop execution
# Use this for final actions or to return results to the user
class ShowResult(StopTool):
    """Show the final result"""

    # This tool doesn't need parameters because it gets data from state
    def execute(self, state):
        # Return the final result that was stored in state by previous tool calls
        return state["result"]


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
    initial_state={"result": None},  # Initialize the agent's state - a shared memory between tools
    verbose=True  # Print detailed logs of what's happening
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
final_state = agent.run(
    model="openai/gpt-4o-mini",  # LLM model to use
    prompt=prompt,               # User's instructions
    system_prompt=system_prompt,  # Instructions for the agent
    budget=10,                   # Maximum number of steps (tool calls) before forced termination
    tools=[Multiply, ShowResult], # List of available tools
)
# When execution completes, final_state contains the agent's final state
# A StopTool will trigger completion, or the agent will stop when budget is exhausted
```

## Core Concepts
Explicit Agent is built around a few simple concepts.

![Explicit Agent Framework](assets/framework.png)



### Agent State

The agent maintains a `state` variable that persists across tool calls. This allows tools to share information, build on previous results, and modify the state itself. The state can be initialized when creating the agent.

### Tool Types

- **`BaseTool`**: This is the base class for creating tools.
- **`StopTool`**: This is the base class for creating stop tools. Stop tools are extremely important because they are the ones that signal when the agent should stop execution.

Both the `BaseTool` and `StopTool` tools can be stateful or stateless based on their `execute` method signature:
- If `execute` method includes a `state` parameter, it's considered stateful (e.g `def execute(state, **kwargs)`)
- If `execute` method doesn't have a `state` parameter, it's considered stateless (e.g `def execute(**kwargs)`)

### Execution Flow

1. The agent receives a prompt from the user
2. The agent generates tool calls based on the prompt and system instructions
3. The tools are executed, potentially updating the agent's state
4. The results are fed back to the agent, which uses them to inform subsequent decisions
5. This continues until a `StopTool` is called or the budget is exhausted

## Examples

For more advanced usage and detailed documentation, see the [examples](examples) directory:

- [Calculator Example](examples/calculator.py): A simple calculator agent that performs arithmetic operations
- [Shopping Cart Example](examples/shopping_cart.py): A more complex example of a shopping assistant that manages a cart
- [PhD Assistant](examples/phd_assistant.py): An advanced example of an agent that helps with academic research and writing

# When is the Explicit Agent Useful?

The Explicit Agent shines in situations where tasks mimic human workflows that involve exploration, decision-making, and adaptability. Here’s a clear definition:
- **Unstructured or Semi-Structured Tasks**: The framework excels when tasks don’t follow a rigid, predefined path and require the agent to adjust its approach based on new information or evolving requirements.

- **Multi-Step Processes with Tool Usage**: It’s ideal for tasks that involve multiple actions or tools, where the agent must decide which tool to use and in what order, depending on the current context.

- **State Management Across Steps**: The agent’s ability to maintain and update a shared state allows it to track progress, share information between tools, and build on previous results.

- **Human-Like Reasoning**: It’s best suited for scenarios where judgment, exploration, or iterative refinement—qualities typically associated with human decision-making—are beneficial.

In short, use this framework when you need an AI agent to autonomously handle complex, real-world problems that require transparency, control, and a sequence of thoughtful steps—much like a human would.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
