from explicit_agent import ExplicitAgent
from explicit_agent.tools import BaseTool, StopTool

# ========= DEFINING TOOLS =========
# Tools are the actions the calculator agent can perform

class Add(BaseTool):
    """Add two numbers"""
    a: int | float  # First number
    b: int | float  # Second number

    @staticmethod
    def execute(state, a, b):
        result = a + b
        state["result"] = result
        return f"Added {a} + {b} = {result}"

class Subtract(BaseTool):
    """Subtract the second number from the first"""
    a: int | float  # First number
    b: int | float  # Second number

    @staticmethod
    def execute(state, a, b):
        result = a - b
        state["result"] = result
        return f"Subtracted {a} - {b} = {result}"

class Multiply(BaseTool):
    """Multiply two numbers"""
    a: int | float  # First number
    b: int | float  # Second number

    @staticmethod
    def execute(state, a, b):
        result = a * b
        state["result"] = result
        return f"Multiplied {a} × {b} = {result}"

class Divide(BaseTool):
    """Divide the first number by the second"""
    a: int | float  # First number
    b: int | float  # Second number

    @staticmethod
    def execute(state, a, b):
        if b == 0:
            return "Error: Cannot divide by zero"
        result = a / b
        state["result"] = result
        return f"Divided {a} ÷ {b} = {result}"

class Power(BaseTool):
    """Raise the first number to the power of the second"""
    base: int | float  # Base number
    exponent: int | float  # Exponent

    @staticmethod
    def execute(state, base, exponent):
        result = base ** exponent
        state["result"] = result
        return f"Calculated {base} ^ {exponent} = {result}"

class SquareRoot(BaseTool):
    """Calculate the square root of a number"""
    number: int | float  # Number to find square root of

    @staticmethod
    def execute(state, number):
        if number < 0:
            return "Error: Cannot calculate square root of a negative number"
        result = number ** 0.5
        state["result"] = result
        return f"Square root of {number} = {result}"

class ShowResult(StopTool):
    """Show the final result and stop execution"""

    @staticmethod
    def execute(state):
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
        system_prompt=system_prompt,
        initial_state={"result": None},
        verbose=True  # Set to True to see detailed logs
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
    final_state = calculator.run(
        model="openai/gpt-4o-mini",  # Or any other supported model
        prompt=calculation_task,
        budget=10,  # Maximum number of tool calls before forced termination
        tools=[Add, Subtract, Multiply, Divide, Power, SquareRoot, ShowResult],
    )

    # final_state will contain the agent's final state after execution
    print(f"Calculation completed with final state: {final_state}")
