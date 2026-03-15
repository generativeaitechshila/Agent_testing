import json
import time
import numpy as np
from openai import OpenAI

# ---------- CONFIG ----------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    #provide you api  key
)

MODEL_NAME = "openai/gpt-4o-mini"

EXTRA_HEADERS = {
    "HTTP-Referer": "https://generativeaitechshila.com",
    "X-Title": "PGI History Evaluator",
}

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

# ---------- FUNCTION: Single Judgment Iteration ----------
def get_llm_judgment(history):
    """
    Runs a single judgment with scientific constraints and credit-safety.
    """
    prompt = f"""
Evaluate the Plan-to-Goal Integrity (PGI) of the following agent history.

### AGENT HISTORY:
{json.dumps(history, indent=2)}

### RUBRIC:
- 0: FAILED. Missing constraints (Landmarks/Cuisine) or tool mismatch.
- 1: PARTIAL. All goals mentioned, but lacks specific tool-backed detail.
- 2: FULL ALIGNMENT. Detailed, logical, and fully supported by tools.

### OUTPUT FORMAT (Strict JSON):
{{
  "evidence": "List user goals and tool outputs found",
  "reasoning": "Step-by-step audit of the plan vs the goal",
  "score": 0, 1, or 2,
  "confidence": 0.0 to 1.0
}}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a scientific QA judge. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1000,  # Prevents 402 Credit Errors
            extra_headers=EXTRA_HEADERS,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        return safe_json_parse(content)
    except Exception as e:
        print(f"⚠️ API Error during iteration: {e}")
        return None

# ---------- FUNCTION: Ensemble Evaluation ----------
def run_ensemble_eval(history, iterations=3):
    judgments = []
    
    # 1. Run multiple iterations (Self-Consistency)
    for i in range(iterations):
        result = get_llm_judgment(history)
        if result:
            judgments.append(result)
        time.sleep(0.5) # Gentle delay for rate limits
        
    # 2. Check for empty results to prevent ZeroDivisionError
    if not judgments:
        return {
            "final_score": 0,
            "variance_std": 0,
            "error": "No valid judgments received from API."
        }
        
    # 3. Weighted Majority Vote (CISC Strategy)
    scores = [float(j.get('score', 0)) for j in judgments]
    weights = [float(j.get('confidence', 0)) for j in judgments]
    
    sum_weights = sum(weights)
    if sum_weights <= 0:
        # Fallback to simple average if all confidence scores are zero
        weighted_score = np.mean(scores)
    else:
        weighted_score = np.average(scores, weights=weights)
    
    final_score = int(round(weighted_score))
    
    # 4. Outlier Detection (Variance Check)
    std_dev = np.std(scores)
    needs_human_review = std_dev > 0.5 
    
    return {
        "final_score": final_score,
        "weighted_avg": round(float(weighted_score), 2),
        "variance_std": round(float(std_dev), 2),
        "needs_human_review": needs_human_review,
        "details": judgments
    }

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    # Mock history based on your chat_001
    sample_history = {
        "id": "chat_001",
        "steps": [
            {"type": "human", "content": "Plan a 3-day itinerary for Paris..."},
            {"type": "ai", "content": "I'll use my itinerary generator..."},
            {"type": "tool_call", "tool": "Itinerary Generator", "output": "Day 1: Eiffel Tower..."},
            {"type": "ai", "content": "Here is your itinerary..."}
        ]
    }

    print("🔬 Running Scientific PGI Evaluation...")
    result = run_ensemble_eval(sample_history, iterations=2)
    
    print("-" * 30)
    print(f"Final Score: {result.get('final_score')}")
    print(f"Confidence Weighted Avg: {result.get('weighted_avg')}")
    print(f"Needs Human Review: {result.get('needs_human_review')}")
    print("-" * 30)