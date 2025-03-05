from openai import OpenAI
from rich.console import Console
import json
from typing import Optional, List, Any, Type
from .tools import register_tools
from .tools import BaseTool, StateAwareTool, StopTool

class ExplicitAgent:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        initial_state: Optional[dict] = None,
    ):
        """
        Initialize the ExplicitAgent with the given API key, base URL, and model.
        This uses the OpenAI API.

        Args:
            `api_key`: The API key for the provider (e.g. OpenAI, OpenRouter, Anthropic, etc.)
            `base_url`: The base URL for the provider (e.g. OpenAI, OpenRouter, Anthropic, etc.)
            `system_prompt`: Optional system prompt to guide the agent's behavior
            `initial_state`: Optional initial state for the agent
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        self.console = Console()
        
        if system_prompt:
            self.messages = [{"role": "system", "content": system_prompt}]
        else:
            self.messages = []
            
        self.state = initial_state or {}

    def _process_tool_calls(self, tool_calls, tools) -> tuple[dict, bool]:
        """
        Process multiple tool calls and execute them sequentially.
        
        Args:
            `tool_calls`: List of tool calls from the LLM response
            `tools`: Dictionary mapping tool classes to their OpenAI definitions
            
        Returns:
            tuple[dict, bool]: (Current state, Whether to stop execution)
        """
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                self._handle_tool_error(tool_call.id, f"Invalid tool arguments: {e}")
                continue

            self.console.print(f"[blue]Tool Call:[/blue] {tool_name}({tool_args})")

            try:
                tool_class = next((tool for tool in tools.keys() if tool.__name__ == tool_name), None)
                if not tool_class:
                    raise ValueError(f"Unknown tool: {tool_name}")
                
                # Handle StopTool
                if issubclass(tool_class, StopTool):
                    result = tool_class.execute(state=self.state, **tool_args)
                    self._append_tool_response(tool_call.id, result)
                    self.console.print("[blue]Agent execution complete[/blue]")
                    return result, True
                
                # Execute the tool based on whether it's state-aware or not
                if issubclass(tool_class, StateAwareTool):
                    # State-aware tool - pass the state
                    result = tool_class.execute(state=self.state, **tool_args)
                    # Update the agent's state with the tool's result
                    self.state = result
                else:
                    # Stateless tool - don't pass the state
                    result = tool_class.execute(**tool_args)
                
                self._append_tool_response(tool_call.id, result)
                self.console.print(f"[blue]Tool Call Result:[/blue] {tool_name}(...) ->\n{result}")

            except Exception as e:
                self._handle_tool_error(tool_call.id, f"Error executing {tool_name}: {e}")
            
        return self.state, False

    def _append_tool_response(self, tool_call_id: str, result: Any) -> None:
        """
        Append a successful tool response to the message history.
        
        Args:
            tool_call_id: The ID of the tool call
            result: The result of the tool execution
        """
        # Ensure the result can be serialized to JSON
        if not isinstance(result, (str, dict, list, int, float, bool)) and result is not None:
            result = str(result)

        tool_call_response = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"result": result})
        }
            
        self.messages.append(tool_call_response)

        self.console.print(f"[blue]Tool Call Raw Response:[/blue] {tool_call_response}")

    def _handle_tool_error(self, tool_call_id: str, error_msg: str) -> None:
        """
        Handle and log tool execution errors.
        
        Args:
            tool_call_id: The ID of the tool call
            error_msg: The error message
        """
        self.console.print(f"[red]{error_msg}[/red]")
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"error": error_msg})
        })

    def run(
        self, 
        model: str,
        prompt: str, 
        budget: int = 20,
        tools: Optional[List[Type[StopTool | BaseTool | StateAwareTool]]] = None,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = False,
    ) -> dict:
        """
        Run the ExplicitAgent with the given prompt and tools.
        
        Args:
            `model`: The model to use for the agent. The model name format depends on the provider specified during initialization (e.g., "gpt-4o-mini" for OpenAI, "openai/gpt-4o-mini" for OpenRouter, etc.)
            `prompt`: The user's request to process
            `tools`: List of tool classes to register or dictionary of already registered tools. If no tools are provided, the agent will act as a simple chatbot.
            `tool_choice`: The tool choice to use for the agent. Can be `"auto"`, `"required"`, or a specific tool (i.e `{"type": "function", "function": {"name": "get_weather"}}`)
            `budget`: The maximum number of steps to run. The agent will stop if it reaches this limit.
            `parallel_tool_calls`: Whether to allow the model to call multiple functions in parallel
            
        Returns:
            dict: The final state of the agent
        """
        # Process tools based on input type
        if not tools:
            tools = {}
            tool_choice = "auto"
        else:
            tools = register_tools(tools)
            
        self.messages.append({"role": "user", "content": prompt})

        current_step = 0

        while True:
            self.console.rule(f"[green]Agent Step {current_step+1} (max steps: {budget})[/green]")
            current_step += 1

            if current_step >= budget:
                self.console.print("[orange]Warning: The agent has reached the maximum budget of steps without completion[/orange]")
                return self.state

            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=self.messages,
                    tools=list(tools.values()),
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )

                if response.choices:
                    message = response.choices[0].message
                    self.messages.append(message)
                    
                    # If no tool calls, just continue the conversation
                    if not message.tool_calls:
                        self.console.print(f"[green]Agent:[/green] {message.content}")
                        return self.state
                    
                    # Process tool calls
                    tool_calls = message.tool_calls
                    self.state, done = self._process_tool_calls(tool_calls=tool_calls, tools=tools)

                    if done:
                        return self.state
                else:
                    raise Exception(f"No response from client: {response}")

            except Exception as e:
                self.console.print(f"[red]Error while running agent: {str(e)}[/red]")
                raise e
