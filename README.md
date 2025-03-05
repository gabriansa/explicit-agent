# Explicit Agent Framework

A minimalist, transparent framework for building AI agents with full user control.

## Why Explicit Agent?

Most agentic frameworks are overengineered with layers of abstraction that obscure what's actually happening. Explicit Agent cuts through the BS to provide:

- **Complete transparency**: No hidden prompts or "magic" under the hood
- **Full control**: You define exactly how your agent behaves
- **Minimal infrastructure**: Only the essentials needed to run capable AI agents
- **Simplicity first**: Build complex behaviors from simple, understandable components

This framework was created to put control back in your hands. It provides the minimum viable infrastructure for running AI agents while maintaining full visibility into their operation.

## Core Concepts

### Agent State

The agent maintains a state dictionary that persists across tool calls and interactions. This allows tools to share information and build on previous results. The state can be initialized when creating the agent and is updated by state-aware tools.

### Tool Types

The framework supports three types of tools:

- **BaseTool**: Standard tools that execute a function and return a result. These tools don't have access to or modify the agent's state.
  
- **StateAwareTool**: Tools that receive the current state and return an updated state. These are perfect for tools that need to read from or write to the agent's persistent memory.
  
- **StopTool**: Special tools that signal when the agent should stop execution. They return a final state with a conclusion and prevent further tool calls.

### Execution Flow

1. The agent receives a prompt from the user
2. The agent generates tool calls based on the prompt and system instructions
3. The tools are executed, potentially updating the agent's state
4. The results are fed back to the agent
5. This continues until a StopTool is called or the budget is exhausted

## Installation

```bash
pip install explicit-agent
```

## Quick Start

```python
import os
from dotenv import load_dotenv
from explicit_agent import ExplicitAgent
from explicit_agent.tools import BaseTool, StateAwareTool, StopTool

# Load API keys from environment
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")

# Define a simple tool
class GetWeather(BaseTool):
    """Get the weather for a location"""
    location: str
    
    @classmethod
    def execute(cls, location: str):
        # In a real implementation, this would call a weather API
        return f"The weather in {location} is sunny and 75°F"

# Define a state-aware tool
class AddToMemory(StateAwareTool):
    """Add information to the agent's memory"""
    key: str
    value: str
    
    @classmethod
    def execute(cls, state: dict, key: str, value: str) -> dict:
        # Update the state with the new memory
        state = state.copy()  # Create a copy to avoid modifying the original
        state.setdefault("memory", {})
        state["memory"][key] = value
        return state

# Define a stopping tool
class Finish(StopTool):
    """Signal that the agent has completed its task"""
    conclusion: str
    
    @classmethod
    def execute(cls, state: dict, conclusion: str) -> dict:
        state = state.copy()
        state["conclusion"] = conclusion
        return state

# Create an agent
agent = ExplicitAgent(
    api_key=api_key,
    base_url=base_url,
    system_prompt="You are a helpful assistant that can retrieve weather information and remember facts."
)

# Run the agent
final_state = agent.run(
    model="openai/gpt-4o-mini",
    prompt="What's the weather in New York? Remember this information for me.",
    tools=[GetWeather, AddToMemory, Finish],
    budget=5  # Maximum number of steps
)

print("Final state:", final_state)
```

## Advanced Usage

For more advanced usage and detailed documentation, see the examples directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
