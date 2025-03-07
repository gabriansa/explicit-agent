from explicit_agent import ExplicitAgent
from explicit_agent.tools import BaseTool, StopTool

from pydantic import Field

# ========= DEFINING TOOLS =========
# Tools for managing a shopping cart

class AddItem(BaseTool):
    """Add an item to the shopping cart"""
    item_name: str  = Field(..., description="The name of the item to add")
    price: float  = Field(..., description="The price of the item")
    quantity: int  = Field(..., description="The quantity of the item")

    def execute(self, state):
        # Initialize the cart if it doesn't exist
        if "cart" not in state:
            state["cart"] = {}
            state["total"] = 0.0
        
        # Add or update the item in the cart
        if self.item_name in state["cart"]:
            existing_quantity = state["cart"][self.item_name]["quantity"]
            state["cart"][self.item_name]["quantity"] = existing_quantity + self.quantity
            state["total"] += self.price * self.quantity
            return f"Updated {self.item_name} quantity to {state['cart'][self.item_name]['quantity']} in the cart"
        else:
            state["cart"][self.item_name] = {
                "price": self.price,
                "quantity": self.quantity
            }
            state["total"] += self.price * self.quantity
            return f"Added {self.quantity} {self.item_name}(s) to the cart at ${self.price:.2f} each"

class RemoveItem(BaseTool):
    """Remove an item from the shopping cart"""
    item_name: str  = Field(..., description="The name of the item to remove")
    
    def execute(self, state):
        if "cart" not in state or self.item_name not in state["cart"]:
            return f"Error: {self.item_name} is not in the cart"
        
        # Calculate the refund amount
        refund = state["cart"][self.item_name]["price"] * state["cart"][self.item_name]["quantity"]
        
        # Remove the item and update the total
        del state["cart"][self.item_name]
        state["total"] -= refund
        
        return f"Removed {self.item_name} from the cart"

class UpdateQuantity(BaseTool):
    """Update the quantity of an item in the cart"""
    item_name: str  = Field(..., description="The name of the item to update")
    new_quantity: int  = Field(..., description="The new quantity of the item")
    
    def execute(self, state):
        if "cart" not in state or self.item_name not in state["cart"]:
            return f"Error: {self.item_name} is not in the cart"
        
        if self.new_quantity <= 0:
            # Remove the item if quantity is zero or negative
            refund = state["cart"][self.item_name]["price"] * state["cart"][self.item_name]["quantity"]
            del state["cart"][self.item_name]
            state["total"] -= refund
            return f"Removed {self.item_name} from the cart (quantity set to {self.new_quantity})"
        
        # Calculate the price difference
        old_quantity = state["cart"][self.item_name]["quantity"]
        price = state["cart"][self.item_name]["price"]
        price_difference = price * (self.new_quantity - old_quantity)
        
        # Update the quantity and total
        state["cart"][self.item_name]["quantity"] = self.new_quantity
        state["total"] += price_difference
        
        return f"Updated {self.item_name} quantity to {self.new_quantity}"

class ApplyDiscount(BaseTool):
    """Apply a discount to the cart total"""
    discount_percentage: float  = Field(..., description="The percentage discount to apply (0-100)")
    
    def execute(self, state):
        if "cart" not in state or not state["cart"]:
            return "Error: Cart is empty"
        
        if self.discount_percentage < 0 or self.discount_percentage > 100:
            return "Error: Discount percentage must be between 0 and 100"
        
        if "discount" in state:
            return "Error: A discount has already been applied"
        
        # Calculate the discount amount
        original_total = state["total"]
        discount_amount = original_total * (self.discount_percentage / 100)
        discounted_total = original_total - discount_amount
        
        # Update the state
        state["discount"] = {
            "percentage": self.discount_percentage,
            "amount": discount_amount
        }
        state["total"] = discounted_total
        
        return f"Applied a {self.discount_percentage}% discount. Saved ${discount_amount:.2f}"

class ShowCart(BaseTool):
    """Display the current contents of the shopping cart"""
    
    def execute(self, state):
        if "cart" not in state or not state["cart"]:
            return "Your cart is empty"
        
        # Build a summary of the cart
        cart_summary = "Current Shopping Cart:\n"
        cart_summary += "------------------------\n"
        
        for item_name, details in state["cart"].items():
            price = details["price"]
            quantity = details["quantity"]
            item_total = price * quantity
            cart_summary += f"{item_name}: {quantity} x ${price:.2f} = ${item_total:.2f}\n"
        
        cart_summary += "------------------------\n"
        
        # Add discount information if applicable
        if "discount" in state:
            discount_percentage = state["discount"]["percentage"]
            discount_amount = state["discount"]["amount"]
            original_total = state["total"] + discount_amount
            cart_summary += f"Subtotal: ${original_total:.2f}\n"
            cart_summary += f"Discount: {discount_percentage}% (-${discount_amount:.2f})\n"
        
        # Add the total
        cart_summary += f"Total: ${state['total']:.2f}"
        
        return cart_summary

class TaskComplete(StopTool):
    """Signal the end of the task"""

    def execute(self):
        return "Task complete"

class Checkout(StopTool):
    """Process the checkout and complete the shopping session"""
    
    def execute(self, state):
        if "cart" not in state or not state["cart"]:
            return "Cannot checkout: Your cart is empty"
        
        # Build a summary of the purchase
        receipt = "Thank you for your purchase!\n"
        receipt += "============================\n"
        receipt += "Receipt:\n"
        
        for item_name, details in state["cart"].items():
            price = details["price"]
            quantity = details["quantity"]
            item_total = price * quantity
            receipt += f"{item_name}: {quantity} x ${price:.2f} = ${item_total:.2f}\n"
        
        receipt += "----------------------------\n"
        
        # Add discount information if applicable
        if "discount" in state:
            discount_percentage = state["discount"]["percentage"]
            discount_amount = state["discount"]["amount"]
            original_total = state["total"] + discount_amount
            receipt += f"Subtotal: ${original_total:.2f}\n"
            receipt += f"Discount: {discount_percentage}% (-${discount_amount:.2f})\n"
        
        # Add the final total
        receipt += f"Total: ${state['total']:.2f}\n"
        receipt += "============================\n"
        receipt += "Payment processed successfully!"
        
        return receipt


# Example of how to use this shopping cart
if __name__ == "__main__":
    # Replace with your actual API key and base URL
    api_key = "your_api_key"
    base_url = "your_base_url"  # For example, "https://api.openai.com/v1"
    
    # System prompt that defines the shopping cart agent
    system_prompt = """
    You are a shopping assistant that helps users manage their shopping cart.
    
    These are the actions you can perform:
    - AddItem: Add an item to the cart with a name, price, and optional quantity
    - RemoveItem: Remove an item completely from the cart
    - UpdateQuantity: Change the quantity of an item in the cart
    - ApplyDiscount: Apply a percentage discount to the entire cart
    - ShowCart: Display the current contents and total of the cart
    - Checkout: Complete the purchase and show the final receipt
    - TaskComplete: Signal the end of the task based on the user's request
    
    When you are done with a user's request, use the `TaskComplete` tool.
    When the user is done shopping and ready to pay, use the `Checkout` tool.
    """
    
    # Initialize the shopping cart agent with empty state
    shopping_agent = ExplicitAgent(
        api_key=api_key,
        base_url=base_url,
        initial_state={},  # Empty initial state
        verbose=True  # Set to True to see detailed logs
    )
    
    # Define individual shopping instructions
    shopping_instructions = [
        "Add a laptop that costs $999.99",
        "Add 3 books at $14.99 each",
        "Add a pair of headphones for $79.95",
        "Show me what's in my cart",
        "I changed my mind - update the book quantity to 2",
        "Apply a 10% discount since I'm a loyal customer",
        "Show me the updated cart",
        "Check out and complete my purchase"
    ]
    
    # Set up available tools
    tools = [AddItem, RemoveItem, UpdateQuantity, ApplyDiscount, ShowCart, Checkout, TaskComplete]
    
    # Run the agent for each instruction separately, preserving state between runs
    current_state = {}  # Initialize empty state
    
    print("Starting interactive shopping session...")
    print("=" * 50)
    
    for i, instruction in enumerate(shopping_instructions):
        print(f"\nStep {i+1}: {instruction}")
        print("-" * 50)
        
        # Update the agent with the current state
        shopping_agent.state = current_state
        
        # Run the agent for just this instruction
        updated_state = shopping_agent.run(
            model="openai/gpt-4o-mini",  # Or any other supported model
            prompt=instruction,
            system_prompt=system_prompt,
            budget=6,  # Smaller budget for each individual instruction
            tools=tools,
        )
        
        # Save the updated state for the next instruction
        current_state = updated_state
        
        # Display the current state of the cart after this instruction
        print("\nCurrent cart state after this instruction:")
        if "cart" in current_state and current_state["cart"]:
            items_in_cart = len(current_state["cart"])
            total = current_state["total"]
            print(f"Items in cart: {items_in_cart}")
            print(f"Current total: ${total:.2f}")
        else:
            print("Cart is empty")
        
        print("=" * 50)
    
    print("\nShopping session completed!")
    print("Final state:", current_state)
