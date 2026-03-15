# run_deepeval_multiple_cases.py
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall

# ---------- CONFIG ----------
AGENT_HISTORY_PATH = Path(r"C:\Generative_AI_Projects\gen_test_x\input_data\agent_history_tc_2.json")
OUTPUT_REPORT_PATH = Path("deepeval_multiple_case_report.json")

# Choose a reliable evaluation model that returns structured JSON.
# If you get "invalid JSON" errors, try "gpt-4o" or another stronger model.
EVAL_MODEL = "gpt-4o"  # change if needed
THRESHOLD = 0.7
INCLUDE_REASON = True


# ---------- HELPERS: load / extract ----------
def load_agent_history(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_tool_calls_with_positions(history: List[Dict[str, Any]]):
    """
    Returns list of dicts {toolcall: ToolCall, invoke_index: int, output_index: Optional[int]}
    where invoke_index is index of agent_action step, output_index is index of tool_output step (if present).
    If tool_output appears before agent_action, it's treated as output-only with invoke_index=None.
    """
    tool_entries = []
    pending = []

    for idx, step in enumerate(history):
        ttype = step.get("type")
        if ttype == "agent_action" and step.get("tool"):
            tc = ToolCall(
                name=step.get("tool"),
                description=step.get("log") or f"Invoked by {step.get('agent')}",
                input_parameters=step.get("tool_input", {}) or {},
                output=""  # fill once we see tool_output
            )
            pending.append({"toolcall": tc, "invoke_index": idx, "output_index": None})

        elif ttype == "tool_output" and step.get("tool"):
            tool_name = step.get("tool")
            output_val = step.get("output", "")
            # match earliest pending invocation with same tool name lacking output
            match_idx = None
            for i, entry in enumerate(pending):
                if entry["toolcall"].name == tool_name and (entry["toolcall"].output is None or entry["toolcall"].output == ""):
                    match_idx = i
                    break
            if match_idx is not None:
                entry = pending.pop(match_idx)
                # attach output (serialize dict/list)
                if isinstance(output_val, (dict, list)):
                    entry["toolcall"].output = json.dumps(output_val, ensure_ascii=False)
                else:
                    entry["toolcall"].output = str(output_val)
                entry["output_index"] = idx
                tool_entries.append(entry)
            else:
                # output-only (no preceding invocation)
                tc = ToolCall(
                    name=tool_name,
                    description=f"Output-only for {tool_name}",
                    input_parameters={},
                    output=json.dumps(output_val, ensure_ascii=False) if isinstance(output_val, (dict, list)) else str(output_val),
                )
                tool_entries.append({"toolcall": tc, "invoke_index": None, "output_index": idx})

    # any remaining pending invocations without outputs: include them with empty output
    for entry in pending:
        tool_entries.append(entry)

    # sort by earliest non-None index (prefer invoke_index, else output_index)
    def sort_key(e):
        return (e["invoke_index"] if e["invoke_index"] is not None else (e["output_index"] if e["output_index"] is not None else float("inf")))
    tool_entries.sort(key=sort_key)
    return tool_entries


def find_human_indices(history: List[Dict[str, Any]]) -> List[int]:
    return [i for i, s in enumerate(history) if s.get("type") == "human"]


def find_agent_answer_in_window(history: List[Dict[str, Any]], start_idx: int, end_idx: int) -> Optional[str]:
    """
    Finds the first agent final answer (agent_finish/agent_final/ai) in history[start_idx:end_idx].
    Returns the content string or None.
    """
    agent_answer_types = ("agent_finish", "agent_final", "ai")
    for j in range(start_idx, end_idx):
        s = history[j]
        if s.get("type") in agent_answer_types and s.get("content"):
            return s["content"]
    return None


def tool_entries_for_window(tool_entries, window_start, window_end):
    """
    Returns a list of ToolCall objects whose invoke_index or output_index lies inside [start, end).
    """
    selected = []
    for e in tool_entries:
        inv = e.get("invoke_index")
        out = e.get("output_index")
        if (inv is not None and window_start <= inv < window_end) or (out is not None and window_start <= out < window_end):
            selected.append(e["toolcall"])
    return selected


# ---------- MAIN: Build and Run Multiple Test Cases ----------
def run_multiple_cases(history_path: Path, expected_output_template: str = None):
    history = load_agent_history(history_path)
    if not isinstance(history, list):
        raise RuntimeError("Agent history JSON must be a top-level list of steps.")

    tool_entries = extract_tool_calls_with_positions(history)
    human_indices = find_human_indices(history)
    total_humans = len(human_indices)

    results = []

    for idx_pos, h_idx in enumerate(human_indices):
        # window: from human step +1 up to next human step (or end)
        start = h_idx + 1
        end = human_indices[idx_pos + 1] if idx_pos + 1 < total_humans else len(history)

        human_text = history[h_idx].get("content", "")
        agent_answer = find_agent_answer_in_window(history, start, end)
        tools_in_window = tool_entries_for_window(tool_entries, start, end)

        # If no explicit expected_output_template provided, we use a simple default tailored to the human task.
        expected_output = expected_output_template or f"Provide an appropriate response to: {human_text}"

        # Build test case (tools_called may be empty list)
        test_case = LLMTestCase(
            input=human_text,
            expected_output=expected_output,
            actual_output=agent_answer or "",
            tools_called=tools_in_window,  # may be empty []
        )

        # Configure metric for each case
        metric = TaskCompletionMetric(
            threshold=THRESHOLD,
            model=EVAL_MODEL,
            include_reason=INCLUDE_REASON,
        )

        # Run metric (synchronous API)
        try:
            metric.measure(test_case)
            score = metric.score
            reason = metric.reason
        except Exception as ex:
            # capture errors (e.g., evaluation model JSON parsing failure)
            score = None
            reason = f"Evaluation failed: {ex}"

        results.append({
            "human_index": idx_pos,
            "human_step_index": h_idx,
            "human_text": human_text,
            "agent_answer": agent_answer,
            "num_tools_used": len(tools_in_window),
            "tool_names": [t.name for t in tools_in_window],
            "score": score,
            "reason": reason,
        })

        # print progress
        print(f"[{idx_pos+1}/{total_humans}] Human idx={h_idx} tools={len(tools_in_window)} score={score}")

    # Save report
    with OUTPUT_REPORT_PATH.open("w", encoding="utf-8") as fout:
        json.dump({"report_for": str(history_path), "cases": results}, fout, indent=2, ensure_ascii=False)

    print(f"\nSaved report to: {OUTPUT_REPORT_PATH}")
    return results


if __name__ == "__main__":
    run_multiple_cases(AGENT_HISTORY_PATH)
