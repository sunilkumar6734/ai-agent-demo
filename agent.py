
import ast
import operator
from pathlib import Path
from ollama import chat

MODEL = "qwen3.5:4b"
NOTES_FILE = Path(__file__).with_name("notes.txt")

# Tool 1: Safe calculator
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def evaluate(node):
    if isinstance(node, ast.Expression):
        return evaluate(node.body)

    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](
            evaluate(node.left),
            evaluate(node.right)
        )

    raise ValueError("Unsupported calculation")

def calculator(expression: str) -> str:
    if len(expression) > 200:
        return "Error: Expression too long"

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree)
        if abs(result) > 1e100:
            return "Error: Result too large"
        return str(result)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        return f"Calculation error: {e}"

# Tool 2: Save a note locally
def save_note(text: str) -> str:
    if not text.strip():
        return "Error: Cannot save an empty note"

    if len(text) > 2000:
        return "Error: Note is too long"

    with NOTES_FILE.open("a", encoding="utf-8") as file:
        file.write(text + "\n")

    return "Note saved successfully"

# Tool 3: Read saved notes
def read_notes() -> str:
    if not NOTES_FILE.exists():
        return "No notes saved yet"

    return NOTES_FILE.read_text(encoding="utf-8")[-5000:] or "No notes saved yet"

# Functions available to the agent
AVAILABLE_TOOLS = {
    "calculator": calculator,
    "save_note": save_note,
    "read_notes": read_notes,
}

# Tool definitions supplied to the model
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a mathematical expression using +, -, *, /.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a text note to a local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_notes",
            "description": "Read previously saved notes.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful local AI agent. "
            "Use the available tools when necessary. "
            "For calculations, always use calculator. "
            "Only save notes when the user explicitly asks. "
            "Never claim a tool succeeded unless its result confirms success."
        )
    }
]

print("Local AI Agent Started!")
print("Commands: exit, /clear\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        break

    if question.lower() == "/clear":
        messages = messages[:1]
        print("Conversation cleared.\n")
        continue

    if not question:
        continue

    messages.append({"role": "user", "content": question})

    try:
        # Agent loop: model chooses tools, Python executes them
        for step in range(5):
            
            response = chat(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                think=False,
                options={"num_ctx": 4096}
            )

            # DEBUG: See what Ollama returned
            print("\n========== OLLAMA RESPONSE ==========")
            print("Role:", response.message.role)
            print("Content:", response.message.content)
            print("Tool calls:", response.message.tool_calls)
            print("=====================================\n")

            messages.append(response.message)

            if not response.message.tool_calls:
                print("\nAI:", response.message.content, "\n")
                break

            for call in response.message.tool_calls:
                name = call.function.name
                args = call.function.arguments

                print(f"[Tool called: {name}]")

                function = AVAILABLE_TOOLS.get(name)

                if function is None:
                    result = f"Unknown tool: {name}"
                else:
                    try:
                        result = function(**args)
                    except Exception as e:
                        result = f"Tool error: {e}"

                print("\n========== TOOL RESPONSE ==========")
                print("Role: tool")
                print("Tool name:", name)
                print("Result:", result)
                print("===================================\n")

                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": str(result)
                })
        else:
            print("Agent reached its tool-step limit.\n")

    except Exception as e:
        print("Error:", e)
