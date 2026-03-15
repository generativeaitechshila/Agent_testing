import json
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric

# ----------------------------------------------------------
# 1️⃣ Load JSON files (Expected and Actual)
# ----------------------------------------------------------
EXPECTED_PATH = r"C:\Generative_AI_Projects\gen_test_x\input_data\processed_data\ground_truth.json"
ACTUAL_PATH = r"C:\Generative_AI_Projects\gen_test_x\input_data\processed_data\extracted_data.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

expected_data = load_json(EXPECTED_PATH)
actual_data = load_json(ACTUAL_PATH)

# ----------------------------------------------------------
# 2️⃣ Extract Tool Calls from JSON
# ----------------------------------------------------------
def extract_tool_calls(data, source="expected"):
    """
    Extracts all tools from router + tools sections and formats as Deepeval ToolCall.
    """
    tool_calls = []

    # From router section (invoked tools)
    for r in data.get("router", []):
        if r.get("action") == "Invoke Tool":
            tool_calls.append(
                ToolCall(
                    name=r.get("tool", ""),
                    description=f"{source.capitalize()} router invoked tool {r.get('tool', '')}",
                    input_parameters=r.get("tool_input", {}),
                    output=""  # output will be captured from 'tools' section
                )
            )

    # From tools section (actual outputs)
    for t in data.get("tools", []):
        tool_name = t.get("tool", "")
        existing_tool = next((tc for tc in tool_calls if tc.name == tool_name), None)
        if existing_tool:
            existing_tool.output = str(t.get("output", ""))
        else:
            tool_calls.append(
                ToolCall(
                    name=tool_name,
                    description=f"{source.capitalize()} direct tool execution for {tool_name}",
                    input_parameters=t.get("tool_input", {}),
                    output=str(t.get("output", ""))
                )
            )

    return tool_calls

expected_tools_1 = extract_tool_calls(expected_data, "expected")
actual_tools_1 = extract_tool_calls(actual_data, "actual")

# ----------------------------------------------------------
# 3️⃣ Build the Test Case
# ----------------------------------------------------------
test_case = LLMTestCase(
    input="Agent system executed several tools as part of a plan.",
    actual_output="Actual tool invocations and their outputs from the agent system.",
    expected_output="Expected tool invocations and outputs based on the planned workflow.",
    tools_called=actual_tools_1,        # what actually happened
    expected_tools=expected_tools_1    # what should have happened
)

# ----------------------------------------------------------
# 4️⃣ Define the Metric (LLM-based Judge)
# ----------------------------------------------------------
metric = ToolCorrectnessMetric()

# ----------------------------------------------------------
# 5️⃣ Run the Metric
# ----------------------------------------------------------
metric.measure(test_case)

print("\n✅ Agent Tool Evaluation Complete")
print("----------------------------------------------------")
print("Score:", metric.score)
print("Reason:", metric.reason)

# ----------------------------------------------------------
# 6️⃣ Optional: Evaluate multiple test cases
# ----------------------------------------------------------
# evaluate(test_cases=[test_case], metrics=[metric])
