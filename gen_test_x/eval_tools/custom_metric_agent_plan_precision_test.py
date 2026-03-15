import json
import time
from openai import OpenAI

# ---------- CONFIG ----------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    # replace with your key,  # replace with your key
)

#MODEL_NAME = "alibaba/tongyi-deepresearch-30b-a3b:free"
MODEL_NAME="openai/gpt-4o-mini"
# MODEL_NAME = "deepseek/deepseek-r1:free"

EXTRA_HEADERS = {
    "HTTP-Referer": "https://generativeaitechshila.com",
    "X-Title": "Agent Plan Precision Evaluator",
}

# ---------- FUNCTION: Load JSON ----------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- FUNCTION: Safe JSON Parse ----------
def safe_json_parse(text):
    try:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        return json.loads(text)
    except Exception:
        return None

# ---------- FUNCTION: Precision Metric via LLM ----------
def calculate_precision_with_llm(actual_plan, expected_plan, retries=2):
    """
    Uses LLM to estimate Precision between expected and actual plan steps.
    Returns: {"precision": float, "true_positives": int, "false_positives": int, "justification": str}
    """
    precision_prompt = f"""
You are an expert AI Plan Evaluator specializing in Precision Scoring.

Compare two agent plans step-by-step:

1. **Expected Plan** — the ground truth sequence of actions.
2. **Actual Plan** — the agent-generated sequence.

Your task:
- Identify which steps in the Actual Plan correctly match or partially match the Expected Plan.
- Ignore minor wording variations (e.g., synonyms or short paraphrases).
- Consider a step correct if its core action and purpose match.
- Penalize extra or irrelevant steps.

Compute:
- True Positives (TP): Steps correctly matching expected ones.
- False Positives (FP): Incorrect or extra steps.
- Precision = TP / (TP + FP)

Return JSON ONLY:
{{
  "precision": float (0.0 - 1.0),
  "true_positives": int,
  "false_positives": int,
  "justification": "brief explanation"
}}

Expected Plan:
{expected_plan}

Actual Plan:
{actual_plan}
"""

    for attempt in range(retries):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Always return valid JSON with numeric precision value."},
                {"role": "user", "content": precision_prompt}
            ],
            temperature=0,
            extra_headers=EXTRA_HEADERS,
        )

        content = response.choices[0].message.content.strip() if response.choices else ""
        parsed = safe_json_parse(content)

        if parsed and "precision" in parsed:
            try:
                parsed["precision"] = float(parsed["precision"])
            except:
                parsed["precision"] = 0.0
            return parsed

        time.sleep(1)

    return {"precision": 0.0, "true_positives": 0, "false_positives": 0, "justification": "Invalid or empty response."}

# ---------- FUNCTION: Evaluate Plans ----------
def evaluate_precision_for_plans(predicted_plan, ground_truth_plan, output_path=r"C:\Generative_AI_Projects\gen_test_x\report\precision_eval_report.json"):
    predicted = load_json(predicted_plan)
    ground_truth = load_json(ground_truth_plan)

    actual_plans = predicted.get("planner", [])
    expected_plans = ground_truth.get("planner", [])

    results = []
    print("🚀 Starting Precision Evaluation for Agent Plans...\n")

    for i, (act, exp) in enumerate(zip(actual_plans, expected_plans), start=1):
        act_goal = act.get("goal", "")
        exp_goal = exp.get("goal", "")

        precision_eval = calculate_precision_with_llm(act_goal, exp_goal)
        results.append({
            "step": i,
            "expected_plan": exp_goal,
            "actual_plan": act_goal,
            "precision": precision_eval["precision"],
            "true_positives": precision_eval["true_positives"],
            "false_positives": precision_eval["false_positives"],
            "justification": precision_eval["justification"]
        })

        print(f"Step {i}: Precision={precision_eval['precision']:.2f} | TP={precision_eval['true_positives']} | FP={precision_eval['false_positives']}")

    # ---------- SUMMARY ----------
    total = len(results)
    avg_precision = sum(r["precision"] for r in results) / total if total else 0
    total_tp = sum(r["true_positives"] for r in results)
    total_fp = sum(r["false_positives"] for r in results)

    summary = {
        "total_cases": total,
        "average_precision": f"{avg_precision:.2f}",
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "details": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Precision evaluation complete. Report saved to {output_path}")
    print(f"📊 Average Precision: {avg_precision:.2f} | Total TP={total_tp}, FP={total_fp}")
    return summary

# ---------- MAIN ----------
if __name__ == "__main__":
    evaluate_precision_for_plans(
        predicted_plan=r"C:\Generative_AI_Projects\gen_test_x\input_data\processed_data\extracted_data.json",
        ground_truth_plan=r"C:\Generative_AI_Projects\gen_test_x\input_data\processed_data\ground_truth.json"
    )
