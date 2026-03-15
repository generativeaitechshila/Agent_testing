import json
import time
import numpy as np
from openai import OpenAI

# ---------- CONFIG ----------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    # replace with your key, 
)

MODEL_NAME = "openai/gpt-4o-mini" # Or "anthropic/claude-3.5-sonnet" for higher kappa consistency

EXTRA_HEADERS = {
    "HTTP-Referer": "https://generativeaitechshila.com",
    "X-Title": "Scientific Agent Plan Precision Evaluator",
}

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
    


def calculate_precision_with_llm(actual_plan, expected_plan, iterations=1):
    """
    Implements Reasoning-First Prompting and CISC (Confidence Score).
    """
    precision_prompt = f"""
    You are a Scientific AI Plan Auditor. Evaluate the Precision of the 'Actual Plan' against the 'Expected Plan'.

    ### EVALUATION STRATEGY (Reasoning-First):
    1. **Evidence Extraction**: Identify core steps in both plans.
    2. **Alignment Check**: Match Actual steps to Expected steps.
    3. **Pointwise Scoring**: 
    - TP: Actual step logically achieves an Expected step.
    - FP: Actual step is redundant, hallucinated, or incorrect.

    ### INPUTS:
    Expected: {expected_plan}
    Actual: {actual_plan}

    ### OUTPUT FORMAT (Strict JSON):
    {{
    "reasoning": "Step-by-step audit of plan alignment",
    "true_positives": int,
    "false_positives": int,
    "precision": float,
    "confidence": float (0.0 to 1.0)
    }}
    """

    judgments = []
    for _ in range(iterations):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a cold, analytical auditor. Output JSON only."},
                    {"role": "user", "content": precision_prompt}
                ],
                temperature=0,
                max_tokens=1000, # Solves the 402 Credit Error
                extra_headers=EXTRA_HEADERS,
            )
            content = response.choices[0].message.content.strip()
            parsed = safe_json_parse(content)
            if parsed:
                judgments.append(parsed)
        except Exception as e:
            print(f"⚠️ API Error: {e}")
            continue # Move to next iteration or exit loop

    # --- HARDENED CALCULATION LOGIC ---
    if not judgments:
        return {
            "precision": 0.0, 
            "true_positives": 0, 
            "false_positives": 0, 
            "confidence": 0.0, 
            "reasoning": "Evaluation failed: No valid response from LLM."
        }

    # Extract scores and weights
    scores = [float(j.get("precision", 0)) for j in judgments]
    weights = [float(j.get("confidence", 0)) for j in judgments]

    # Check if weights are valid to avoid ZeroDivisionError
    sum_weights = sum(weights)
    if sum_weights <= 0:
        # Fallback: Use simple average if all confidence scores are 0
        avg_precision = np.mean(scores)
    else:
        avg_precision = np.average(scores, weights=weights)
    
    # Return the best available judgment
    best_idx = np.argmax(weights) if sum_weights > 0 else 0
    final_judgment = judgments[best_idx]
    final_judgment["precision"] = round(float(avg_precision), 2)
    
    return final_judgment



# ---------- UPDATED EVALUATION LOOP ----------
def evaluate_precision_for_plans(predicted_plan_path, ground_truth_plan_path, output_path="C:\Generative_AI_Projects\gen_test_x\report\scientific_eval_report.json"):
    with open(predicted_plan_path, "r") as f: predicted = json.load(f)
    with open(ground_truth_plan_path, "r") as f: ground_truth = json.load(f)

    actual_steps = predicted.get("planner", [])
    expected_steps = ground_truth.get("planner", [])

    results = []
    for i, (act, exp) in enumerate(zip(actual_steps, expected_steps), start=1):
        # Extract the content/goal for comparison
        act_content = act.get("goal", act.get("content", ""))
        exp_content = exp.get("goal", exp.get("content", ""))

        eval_data = calculate_precision_with_llm(act_content, exp_content)
        
        results.append({
            "step": i,
            "precision": eval_data["precision"],
            "confidence": eval_data.get("confidence", 0),
            "reasoning": eval_data.get("reasoning", ""),
            "tp": eval_data["true_positives"],
            "fp": eval_data["false_positives"]
        })
        print(f"Step {i}: Precision={eval_data['precision']} (Conf: {eval_data.get('confidence')})")

    # Save summary
    summary = {
        "avg_precision": np.mean([r["precision"] for r in results]),
        "avg_confidence": np.mean([r["confidence"] for r in results]),
        "details": results
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Report saved to {output_path}")

if __name__ == "__main__":
    # Ensure paths exist before running
    evaluate_precision_for_plans(
        predicted_plan_path=r"C:\Generative_AI_Projects\gen_test_x\input_data\processed_data\extracted_data.json",
        ground_truth_plan_path=r"C:\Generative_AI_Projects\gen_test_x\input_data\processed_data\ground_truth.json"
    )