# Explicit Agent Framework

A framework for creating explicit tool-using agents.

## Features

- Create agents that use tools in an explicit, controllable manner
- Support for OpenAI compatible APIs (including OpenRouter)
- State management across tool calls

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
