import json
from typing import Optional, List, Any, Type, Dict

from openai import OpenAI
from rich.console import Console
import logging
import traceback

from .tools import register_tools, StopTool, BaseTool


class ExplicitAgent:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        initial_state: Optional[Any] = {},
        verbose: bool = True,
    ):
        """
        Initialize the ExplicitAgent with the given API key, base URL, and model.
        This uses the OpenAI API.

        Args:
            `api_key`: The API key for the provider (e.g. OpenAI, OpenRouter, Anthropic, etc.).
            `base_url`: The base URL for the provider (e.g. OpenAI, OpenRouter, Anthropic, etc.).
            `initial_state`: Optional initial state for the agent. Must be a mutable object (e.g., dict, list).
            `verbose`: Whether to print verbose output to the console (default: True).
        """

        if initial_state is not None:
            try:
                hash(initial_state)
                raise ValueError(
                    f"Initial state must be mutable. Got immutable type: {type(initial_state)}. "
                    "Use a mutable type like dict, list, or a custom class instead."
                )
            except TypeError:
                pass

        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI client: {e}")
            raise

        self.console = Console()
        self.logger = logging.getLogger("explicit_agent")
        self.verbose = verbose

        self.messages: List[Dict[str, Any]] = []

        self.state: Any = initial_state

    def _process_tool_calls(self, tool_calls, tools) -> bool:
        """
        Process multiple tool calls and execute them sequentially.

        Args:
            `tool_calls`: List of tool calls from the LLM response.
            `tools`: Dictionary mapping tool classes to their OpenAI definitions.

        Returns:
            `bool`: Whether to stop execution of the agent.
        """
        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            tool_class = next(
                (tool for tool in tools.keys() if tool.__name__ == tool_name), None
            )

            if not tool_class:
                self._handle_tool_error(
                    tool_call.id,
                    f"Unknown tool: '{tool_name}'"
                )
                continue

            try:
                tool_args = json.loads(tool_call.function.arguments)
                tool_instance = tool_class(**tool_args)
            except Exception as e:
                self._handle_tool_error(
                    tool_call.id,
                    f"Invalid tool arguments for '{tool_name}' tool.\nGiven arguments: {tool_call.function.arguments}\nError: {e}",
                )
                continue

            self.logger.info(f"Tool Call: {tool_name}({tool_args})")
            if self.verbose:
                self.console.print(
                    f"[bold blue]Tool Call:[/bold blue] {tool_name}({tool_args})"
                )

            try:
                is_stop_tool = issubclass(tool_class, StopTool)
                is_stateful = tool_class.is_stateful()

                if is_stateful:
                    result = tool_instance.execute(state=self.state)
                else:
                    result = tool_instance.execute()

                try:
                    serialized_result = json.dumps({"result": result})
                except (TypeError, ValueError) as e:
                    self.logger.warning(
                        f"Non-serializable result type: {type(result)}. Converting to string."
                    )
                    serialized_result = json.dumps({"result": str(result)})

                tool_call_response = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": serialized_result,
                }

                self.messages.append(tool_call_response)

                self.logger.info(f"Tool Call Result: {tool_name}(...) -> {result}")
                if self.verbose:
                    self.console.print(
                        f"[bold blue]Tool Call Result:[/bold blue] {tool_name}(...) -> {result}"
                    )

                self.logger.info(f"State: {self.state}")
                if self.verbose and self.state:
                    self.console.print(f"[bold blue]State:[/bold blue] {self.state}")

                if is_stop_tool:
                    self.logger.info("Agent execution complete")
                    if self.verbose:
                        self.console.print(
                            "[bold bright_green]Agent execution complete[/bold bright_green]"
                        )
                    return True

            except Exception as e:
                self._handle_tool_error(
                    tool_call.id,
                    f"Error executing '{tool_name}' tool: {str(e)}",
                )
                continue

        return False

    def _handle_tool_error(
        self, tool_call_id: str, error_msg: str
    ) -> None:
        """
        Handle and log tool execution errors.

        Args:
            `tool_call_id`: The ID of the tool call.
            `error_msg`: The error message.
        """

        error_traceback = traceback.format_exc()
        self.logger.error(error_traceback)
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
        system_prompt: Optional[str] = None,
        budget: int = 20,
        tools: Optional[List[Type[BaseTool]]] = None,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Run the ExplicitAgent with the given prompt and tools.

        Args:
            `model`: The model to use for the agent. The model name format depends on the provider specified during initialization (e.g., "gpt-4o-mini" for OpenAI, "openai/gpt-4o-mini" for OpenRouter, etc.).
            `prompt`: The user's request to process.
            `system_prompt`: Optional system prompt to guide the agent's behavior. If provided, it will replace any existing system prompt in the messages.
            `budget`: The maximum number of steps to run. The agent will stop if it reaches this limit.
            `tools`: List of tool classes. If no tools are provided, the agent will act as a simple chatbot.
            `tool_choice`: The tool choice to use for the agent. Can be `"auto"`, `"required"`, or a specific tool (i.e `{"type": "function", "function": {"name": "get_weather"}}`).
            `parallel_tool_calls`: Whether to allow the model to call multiple functions in parallel.
            `**kwargs`: Additional keyword arguments to pass to the OpenAI API (i.e. `temperature`, `max_tokens`, `reasoning_effort`, etc.).
        Returns:
            `Any`: The final state of the agent.
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
            for tool in tools:
                if not issubclass(tool, BaseTool):
                    raise ValueError(
                        f"Tool class '{tool.__name__}' must be a subclass of BaseTool or StopTool"
                    )
            tools = register_tools(tools)

        if system_prompt:
            self.messages = [msg for msg in self.messages if msg.get("role") != "system"]
            self.messages.insert(0, {"role": "system", "content": system_prompt})

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
                    **kwargs,
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
                error_msg = f"Error while running agent: {str(e)}"
                self.logger.error(f"Error while running agent: {traceback.format_exc()}")
                raise Exception(error_msg)
