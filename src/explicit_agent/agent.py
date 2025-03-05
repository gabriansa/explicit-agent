import json
from typing import Optional, List, Any, Type, Dict, Union

from openai import OpenAI
from rich.console import Console
import logging

from .tools import register_tools
from .tools import StatelessTool, StatefulTool, StopStatelessTool, StopStatefulTool


class ExplicitAgent:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        initial_state: Optional[Any] = None,
        verbose: bool = True,
    ):
        """
        Initialize the ExplicitAgent with the given API key, base URL, and model.
        This uses the OpenAI API.

        Args:
            `api_key`: The API key for the provider (e.g. OpenAI, OpenRouter, Anthropic, etc.)
            `base_url`: The base URL for the provider (e.g. OpenAI, OpenRouter, Anthropic, etc.)
            `system_prompt`: Optional system prompt to guide the agent's behavior
            `initial_state`: Optional initial state for the agent
            `verbose`: Whether to print verbose output to the console (default: True)
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.console = Console()
        self.logger = logging.getLogger("explicit_agent")
        self.verbose = verbose

        if system_prompt:
            self.messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt}
            ]
        else:
            self.messages: List[Dict[str, Any]] = []

        self.state: Any = initial_state

    def _process_tool_calls(self, tool_calls, tools) -> bool:
        """
        Process multiple tool calls and execute them sequentially.

        Args:
            `tool_calls`: List of tool calls from the LLM response
            `tools`: Dictionary mapping tool classes to their OpenAI definitions

        Returns:
            `bool`: Whether to stop execution of the agent
        """
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                self._handle_tool_error(tool_call.id, f"Invalid tool arguments: {e}")
                continue

            self.logger.info(f"Tool Call: {tool_name}({tool_args})")
            if self.verbose:
                self.console.print(
                    f"[bold blue]Tool Call:[/bold blue] {tool_name}({tool_args})"
                )

            try:
                tool_class = next(
                    (tool for tool in tools.keys() if tool.__name__ == tool_name), None
                )
                if not tool_class:
                    raise ValueError(f"Unknown tool: {tool_name}")

                if issubclass(tool_class, StopStatelessTool):
                    result = tool_class.execute(**tool_args)
                    self._append_tool_response(tool_call.id, result)
                    self.logger.info("Agent execution complete")
                    if self.verbose:
                        self.console.print(
                            "[bold bright_green]Agent execution complete[/bold bright_green]"
                        )
                    return True

                if issubclass(tool_class, StopStatefulTool):
                    result = tool_class.execute(state=self.state, **tool_args)
                    self._append_tool_response(tool_call.id, result)
                    self.logger.info("Agent execution complete")
                    if self.verbose:
                        self.console.print(
                            "[bold bright_green]Agent execution complete[/bold bright_green]"
                        )
                    return True

                if issubclass(tool_class, StatefulTool):
                    result = tool_class.execute(state=self.state, **tool_args)

                if issubclass(tool_class, StatelessTool):
                    result = tool_class.execute(**tool_args)

                self._append_tool_response(tool_call.id, result)
                self.logger.info(f"Tool Call Result: {tool_name}(...) -> {result}")
                if self.verbose:
                    self.console.print(
                        f"[bold blue]Tool Call Result:[/bold blue] {tool_name}(...) -> {result}"
                    )

            except Exception as e:
                self._handle_tool_error(
                    tool_call.id, f"Error executing {tool_name}: {e}"
                )

        return False

    def _append_tool_response(self, tool_call_id: str, result: Any) -> None:
        """
        Append a successful tool response to the message history.

        Args:
            `tool_call_id`: The ID of the tool call
            `result`: The result of the tool execution
        """
        try:
            serialized_result = json.dumps({"result": result})
        except (TypeError, ValueError) as e:
            self.logger.warning(
                f"Non-serializable result type: {type(result)}. Converting to string."
            )
            serialized_result = json.dumps({"result": str(result)})

        tool_call_response = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": serialized_result,
        }

        self.messages.append(tool_call_response)

    def _handle_tool_error(self, tool_call_id: str, error_msg: str) -> None:
        """
        Handle and log tool execution errors.

        Args:
            `tool_call_id`: The ID of the tool call
            `error_msg`: The error message
        """
        self.logger.error(error_msg)
        if self.verbose:
            self.console.print(f"[bold red1]{error_msg}[/bold red1]")
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"error": error_msg}),
            }
        )

    def run(
        self,
        model: str,
        prompt: str,
        budget: int = 20,
        tools: Optional[
            List[
                Type[
                    Union[
                        StatelessTool, StatefulTool, StopStatelessTool, StopStatefulTool
                    ]
                ]
            ]
        ] = None,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = False,
    ) -> Any:
        """
        Run the ExplicitAgent with the given prompt and tools.

        Args:
            `model`: The model to use for the agent. The model name format depends on the provider specified during initialization (e.g., "gpt-4o-mini" for OpenAI, "openai/gpt-4o-mini" for OpenRouter, etc.)
            `prompt`: The user's request to process
            `tools`: List of tool classes. If no tools are provided, the agent will act as a simple chatbot.
            `tool_choice`: The tool choice to use for the agent. Can be `"auto"`, `"required"`, or a specific tool (i.e `{"type": "function", "function": {"name": "get_weather"}}`)
            `budget`: The maximum number of steps to run. The agent will stop if it reaches this limit.
            `parallel_tool_calls`: Whether to allow the model to call multiple functions in parallel

        Returns:
            `Any`: The final state of the agent
        """

        if not model or not isinstance(model, str):
            raise ValueError("Model name must be a non-empty string")

        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        if not isinstance(budget, int) or budget <= 0:
            raise ValueError("Budget must be a positive integer")

        if tool_choice not in ["auto", "required"] and not isinstance(
            tool_choice, dict
        ):
            raise ValueError(
                "Tool choice must be 'auto', 'required', or a specific tool configuration"
            )

        if not tools:
            tools = {}
            tool_choice = "auto"
        else:
            valid_base_classes = (
                StatelessTool,
                StatefulTool,
                StopStatelessTool,
                StopStatefulTool,
            )
            for tool in tools:
                if not any(
                    issubclass(tool, base_class) for base_class in valid_base_classes
                ):
                    raise ValueError(
                        f"Tool class '{tool.__name__}' must be a subclass of StatelessTool, StatefulTool, StopStatelessTool, or StopStatefulTool"
                    )
            tools = register_tools(tools)

        self.messages.append({"role": "user", "content": prompt})

        current_step = 0

        while True:
            current_step += 1

            self.logger.info(f"Agent Step {current_step}/{budget}")
            if self.verbose:
                self.console.rule(
                    f"[bold green_yellow]Agent Step {current_step}/{budget}[/bold green_yellow]",
                    style="green_yellow",
                )

            if current_step >= budget:
                warning_msg = f"Warning: The agent has reached the maximum budget of steps without completion (budget: {budget})"
                self.logger.warning(warning_msg)
                if self.verbose:
                    self.console.print(f"[bold orange1]{warning_msg}[/bold orange1]")
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

                    if not message.tool_calls:
                        self.logger.info(f"Agent Message: {message.content}")
                        if self.verbose:
                            self.console.print(
                                f"[bold blue]Agent Message:[/bold blue] {message.content}"
                            )
                        continue

                    if not parallel_tool_calls and len(message.tool_calls) > 1:
                        self.logger.warning(
                            f"Received {len(message.tool_calls)} tool calls when parallel_tool_calls=False. Processing only the first one."
                        )
                        if self.verbose:
                            self.console.print(
                                f"[bold yellow]Warning: Received {len(message.tool_calls)} tool calls when parallel_tool_calls=False. Processing only the first one.[/bold yellow]"
                            )
                        message = message.model_copy(
                            update={"tool_calls": [message.tool_calls[0]]}
                        )

                    self.messages.append(message)

                    tool_calls = message.tool_calls
                    done = self._process_tool_calls(tool_calls=tool_calls, tools=tools)

                    if done:
                        return self.state
                else:
                    error_msg = f"No response from client: {response}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)

            except Exception as e:
                self.logger.error(f"Error while running agent: {str(e)}")
                if self.verbose:
                    self.console.print(
                        f"[bold red1]Error while running agent: {str(e)}[/bold red1]"
                    )
                raise e
