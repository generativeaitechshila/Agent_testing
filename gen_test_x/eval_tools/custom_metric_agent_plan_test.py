import json
import time
from openai import OpenAI

# ---------- CONFIG ----------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    # replace with your key,  # replace with your key
)

MODEL_NAME = "alibaba/tongyi-deepresearch-30b-a3b:free"
# MODEL_NAME = "deepseek/deepseek-r1:free"

EXTRA_HEADERS = {
    "HTTP-Referer": "https://generativeaitechshila.com",
    "X-Title": "Generative AI Plan Evaluator",
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


# ---------- FUNCTION: LLM Judge ----------
def evaluate_plan_with_llm(actual_goal, expected_goal, retries=2):
    """
    Use LLM via OpenRouter to compare plans.
    Returns a dict: {"status": "PASS"/"FAIL", "justification": str}
    """
    prompt = f"""
You are an expert LLM-as-a-Judge for Agent Plan Evaluation.

Compare two plans:
1. **Expected Plan** — the correct sequence of steps.
2. **Actual Plan** — the agent-generated sequence.

Rules:
- Minor wording or ordering variations are acceptable if meaning is preserved.
- Mark FAIL only if key steps are missing, incorrect, or clearly out of order.

Return STRICTLY in JSON:
{{
  "status": "PASS" or "FAIL",
  "justification": "Brief reason if FAIL, else 'All good'."
}}

Expected Plan:
{expected_goal}

Actual Plan:
{actual_goal}
"""

    for attempt in range(retries):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            extra_headers=EXTRA_HEADERS,
        )

        content = response.choices[0].message.content.strip() if response.choices else ""
        parsed = safe_json_parse(content)

        if parsed and "status" in parsed:
            parsed["status"] = parsed["status"].upper().strip()
            if parsed["status"] not in ["PASS", "FAIL"]:
                parsed["status"] = "FAIL"
            return parsed

        time.sleep(1)  # wait before retry

    # fallback
    return {"status": "FAIL", "justification": f"Invalid or empty JSON: {content[:80]}..."}


# ---------- FUNCTION: Evaluate Planner ----------
def evaluate_planner(predicted_plan, ground_truth_plan, output_path=r"C:\Generative_AI_Projects\gen_test_x\report\planner_eval_report.json"):
    predicted = load_json(predicted_plan)
    ground_truth = load_json(ground_truth_plan)

    actual_plans = predicted.get("planner", [])
    expected_plans = ground_truth.get("planner", [])

    results = []
    for i, (act, exp) in enumerate(zip(actual_plans, expected_plans), start=1):
        act_goal = act.get("goal", "")
        exp_goal = exp.get("goal", "")
        evaluation = evaluate_plan_with_llm(act_goal, exp_goal)
        results.append({
            "step": i,
            "expected_goal": exp_goal,
            "actual_goal": act_goal,
            "status": evaluation["status"],
            "justification": evaluation["justification"]
        })
        print(f"Step {i}: {evaluation['status']} | {evaluation['justification']}")

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    summary = {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{(passed / total * 100):.1f}%",
        "details": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ LLM evaluation complete. Report saved to {output_path}")
    print(f"📊 Pass Rate: {summary['pass_rate']} ({passed}/{total} passed)")

    return summary


# ---------- MAIN ----------
if __name__ == "__main__":
    evaluate_planner(
        predicted_plan=r"C:\Generative_AI_Projects\gen_test_x\input_data\processed_data\extracted_data.json",
        ground_truth_plan=r"C:\Generative_AI_Projects\gen_test_x\input_data\processed_data\ground_truth.json"
    )
