import json
from pathlib import Path

def extract_agent_data(agent_history_input):
    planner, router, tools = [], [], []
    step_count = 0

    # --- FIX: Ensure we are iterating over the LIST of events ---
    # If the JSON is {"steps": [...]}, we want the list inside "steps"
    if isinstance(agent_history_input, dict):
        # Try common keys, fallback to the values if no key matches
        agent_history = agent_history_input.get("steps") or agent_history_input.get("history")
        if agent_history is None:
            raise ValueError("❌ JSON dictionary must contain a 'steps' or 'history' list.")
    else:
        agent_history = agent_history_input

    for event in agent_history:
        # Extra safety check
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        agent = event.get("agent") or event.get("type") # Fallback for your specific schema

        # 1️⃣ Planner / AI Reasoning extraction
        # Added "ai" type to match your previous chat history example
        if event_type in ["agent_plan", "ai"]:
            step_count += 1
            planner.append({
                "step": step_count,
                "agent": agent,
                "goal": event.get("content")
            })

        # 2️⃣ Router / Action extraction
        elif event_type == "agent_action" and agent == "RouterAgent":
            step_count += 1
            route_entry = {
                "step": step_count,
                "router": "RouterAgent",
                "log": event.get("log"),
            }

            if "route_to" in event:
                route_entry["action"] = "Route to Sub-Agent"
                route_entry["route_to"] = event["route_to"]
            elif "tool" in event:
                route_entry["action"] = "Invoke Tool"
                route_entry["tool"] = event["tool"]
                route_entry["tool_input"] = event.get("tool_input")

            router.append(route_entry)

        # 3️⃣ Tool usage extraction (Matches "tool_call" or "tool_output")
        elif event_type in ["tool_output", "tool_call"]:
            tools.append({
                "tool": event.get("tool") or event.get("id"), # Handle different schema IDs
                "output": event.get("output") or event.get("content")
            })

    return {"planner": planner, "router": router, "tools": tools}


# -----------------
# Load JSON safely
# -----------------
input_path = Path(r"C:\Generative_AI_Projects\gen_test_x\input_data\agent_history_tc_1.json")

if not input_path.exists():
    raise FileNotFoundError(f"❌ Error: File not found at {input_path}")

if input_path.stat().st_size == 0:
    raise ValueError(f"❌ Error: JSON file '{input_path}' is empty.")

with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Extract data using the fixed function
structured_data = extract_agent_data(raw_data)

# Save structured output
output_path = Path("extracted_data.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(structured_data, f, indent=2, ensure_ascii=False)

print(f"✅ Extraction complete! Found {len(structured_data['planner'])} plan steps.")
print(f"📂 Saved to: {output_path.resolve()}")