import json
import time
from openai import OpenAI

# ---------- CONFIG ----------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    # replace with your key,  # replace with your OpenRouter key
)

#MODEL_NAME = "alibaba/tongyi-deepresearch-30b-a3b:free"
#MODEL_NAME = "deepseek/deepseek-r1:free"
#MODEL_NAME="mistralai/mistral-7b-instruct"
#MODEL_NAME="google/gemini-pro-1.5"
MODEL_NAME="openai/gpt-4o-mini"

EXTRA_HEADERS = {
    "HTTP-Referer": "https://generativeaitechshila.com",
    "X-Title": "Agent Plan Recall Evaluator",
}

# ---------- FUNCTION: Load JSON ----------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- FUNCTION: Safe JSON Parse ----------
def safe_json_parse(text):
    """
    Extract valid JSON substring and parse it safely.
    """
    try:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        return json.loads(text)
    except Exception:
        return None


# ---------- FUNCTION: Recall Metric via LLM ----------
def calculate_recall_with_llm(actual_plan, expected_plan, retries=2):
    """
    Uses LLM to estimate Recall between expected and actual plan steps.
    Returns: {"recall": float, "true_positives": int, "false_negatives": int, "justification": str}
    """
    recall_prompt = f"""
You are an expert AI Plan Evaluator specializing in Recall Scoring.

Compare two agent plans step-by-step:

1. **Expected Plan** — the correct (ground truth) sequence of steps.
2. **Actual Plan** — the agent-generated sequence.

Your task:
- Identify which expected steps are correctly covered or partially covered by the Actual Plan.
- Ignore small wording variations (e.g., synonyms, tense, or phrasing differences).
- A step is considered *covered* if its intent and action align with the expected plan.
- Penalize missing steps that were not addressed in the Actual Plan.

Compute:
- True Positives (TP): Expected steps that were correctly covered by the Actual Plan.
- False Negatives (FN): Expected steps that were missing or not addressed.
- Recall = TP / (TP + FN)

Return JSON ONLY:
{{
  "recall": float (0.0 - 1.0),
  "true_positives": int,
  "false_negatives": int,
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
                {"role": "system", "content": "Always return valid JSON with numeric recall value."},
                {"role": "user", "content": recall_prompt}
            ],
            temperature=0,
            extra_headers=EXTRA_HEADERS,
        )

        content = response.choices[0].message.content.strip() if response.choices else ""
        parsed = safe_json_parse(content)

        if parsed and "recall" in parsed:
            try:
                parsed["recall"] = float(parsed["recall"])
            except:
                parsed["recall"] = 0.0
            return parsed

        time.sleep(1)

    return {"recall": 0.0, "true_positives": 0, "false_negatives": 0, "justification": "Invalid or empty response."}


# ---------- FUNCTION: Evaluate Plans ----------
def evaluate_recall_for_plans(predicted_plan, ground_truth_plan, output_path=r"C:\Generative_AI_Projects\gen_test_x\report\recall_eval_report.json"):
    """
    Compare two sets of plans and compute recall per step and overall average.
    """
    predicted = load_json(predicted_plan)
    ground_truth = load_json(ground_truth_plan)

    actual_plans = predicted.get("planner", [])
    expected_plans = ground_truth.get("planner", [])

    results = []
    print("🚀 Starting Recall Evaluation for Agent Plans...\n")

    for i, (act, exp) in enumerate(zip(actual_plans, expected_plans), start=1):
        act_goal = act.get("goal", "")
        exp_goal = exp.get("goal", "")

        recall_eval = calculate_recall_with_llm(act_goal, exp_goal)
        results.append({
            "step": i,
            "expected_plan": exp_goal,
            "actual_plan": act_goal,
            "recall": recall_eval["recall"],
            "true_positives": recall_eval["true_positives"],
            "false_negatives": recall_eval["false_negatives"],
            "justification": recall_eval["justification"]
        })

        print(f"Step {i}: Recall={recall_eval['recall']:.2f} | TP={recall_eval['true_positives']} | FN={recall_eval['false_negatives']}")

    # ---------- SUMMARY ----------
    total = len(results)
    avg_recall = sum(r["recall"] for r in results) / total if total else 0
    total_tp = sum(r["true_positives"] for r in results)
    total_fn = sum(r["false_negatives"] for r in results)

    summary = {
        "total_cases": total,
        "average_recall": f"{avg_recall:.2f}",
        "total_true_positives": total_tp,
        "total_false_negatives": total_fn,
        "details": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Recall evaluation complete. Report saved to {output_path}")
    print(f"📊 Average Recall: {avg_recall:.2f} | Total TP={total_tp}, FN={total_fn}")
    return summary


# ---------- MAIN ----------
if __name__ == "__main__":
    evaluate_recall_for_plans(
        predicted_plan=r"C:\Generative_AI_Projects\gen_test_x\input_data\processed_data\extracted_data.json",
        ground_truth_plan=r"C:\Generative_AI_Projects\gen_test_x\input_data\processed_data\ground_truth.json"
    )
