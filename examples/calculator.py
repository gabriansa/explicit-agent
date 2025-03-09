from explicit_agent import ExplicitAgent
from explicit_agent.tools import BaseTool, StopTool

from pydantic import Field


# ========= DEFINING STATE =========
# State is the context in which the calculator agent operates
state = {}

# ========= DEFINING TOOLS =========
# Tools are the actions the calculator agent can perform

class Add(BaseTool):
    """Add two numbers"""
    a: int | float  = Field(..., description="The first number to add")
    b: int | float  = Field(..., description="The second number to add")

    def execute(self):
        print(f"Adding {self.a} + {self.b}")
        result = self.a + self.b
        state["result"] = result
        return f"Added {self.a} + {self.b} = {result}"

class Subtract(BaseTool):
    """Subtract the second number from the first"""
    a: int | float  = Field(..., description="The first number")
    b: int | float  = Field(..., description="The second number")

    def execute(self):
        result = self.a - self.b
        state["result"] = result
        return f"Subtracted {self.a} - {self.b} = {result}"

class Multiply(BaseTool):
    """Multiply two numbers"""
    a: int | float  = Field(..., description="The first number")
    b: int | float  = Field(..., description="The second number")

    def execute(self):
        result = self.a * self.b
        state["result"] = result
        return f"Multiplied {self.a} × {self.b} = {result}"

class Divide(BaseTool):
    """Divide the first number by the second"""
    a: int | float  = Field(..., description="The first number")
    b: int | float  = Field(..., description="The second number")

    def execute(self):
        if self.b == 0:
            return "Error: Cannot divide by zero"
        result = self.a / self.b
        state["result"] = result
        return f"Divided {self.a} ÷ {self.b} = {result}"

class Power(BaseTool):
    """Raise the first number to the power of the second"""
    base: int | float  = Field(..., description="The base number")
    exponent: int | float  = Field(..., description="The exponent")

    def execute(self):
        result = self.base ** self.exponent
        state["result"] = result
        return f"Calculated {self.base} ^ {self.exponent} = {result}"

class SquareRoot(BaseTool):
    """Calculate the square root of a number"""
    number: int | float  = Field(..., description="The number to find square root of")

    def execute(self):
        if self.number < 0:
            return "Error: Cannot calculate square root of a negative number"
        result = self.number ** 0.5
        state["result"] = result
        return f"Square root of {self.number} = {result}"

class ShowResult(StopTool):
    """Show the final result and stop execution"""

    def execute(self):
        if "result" not in state or state["result"] is None:
            return "No result has been calculated yet."
        return f"Final result: {state['result']}"


# Example of how to use this calculator
if __name__ == "__main__":
    # Replace with your actual API key and base URL
    api_key = "your_api_key"
    base_url = "your_base_url"  # For example, "https://api.openai.com/v1"

    # System prompt that defines the calculator agent
    system_prompt = """
    You are a calculator assistant.
    These are the operations you can perform:
    - Add: Add two numbers
    - Subtract: Subtract the second number from the first
    - Multiply: Multiply two numbers
    - Divide: Divide the first number by the second
    - Power: Raise the first number to the power of the second
    - SquareRoot: Calculate the square root of a number
    - ShowResult: Display the final result and finish the calculation

    When you are done with all calculations, use the `ShowResult` tool to display the final result.
    """

    # Initialize the calculator agent
    calculator = ExplicitAgent(
        api_key=api_key,
        base_url=base_url,
        verbose=True
    )

    # Example calculation task
    calculation_task = """
    Please perform the following calculations:
    1. Start with 10
    2. Add 5 to it
    3. Multiply the result by 2
    4. Subtract 7
    5. Divide by 3
    6. Show the final result
    """

    # Run the calculator agent
    calculator.run(
        model="openai/gpt-4o-mini",  # Or any other supported model
        prompt=calculation_task,
        system_prompt=system_prompt,
        budget=10,  # Maximum number of tool calls before forced termination
        tools=[Add, Subtract, Multiply, Divide, Power, SquareRoot, ShowResult],
    )

    # state will contain the agent's final state after execution
    print(f"Calculation completed with final state: {state}")
