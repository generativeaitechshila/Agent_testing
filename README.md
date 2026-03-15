This README is designed for a technical audience (AI QA Engineers/MLOps) and highlights the research-backed "Scientific" approach to evaluation we've built.

---

# 🚀 Agent-Eval Framework

### *From "Vibey" to Scientific LLM Evaluations*

This framework provides a robust, research-backed pipeline to evaluate AI Agent performance, specifically focusing on **Plan-to-Goal Integrity (PGI)** and **Execution Precision**. By utilizing **Ensemble Judging** and **Confidence-Informed Self-Consistency (CISC)**, it reduces the stochastic noise typically found in LLM-based evaluations.

---

## 🏗️ Core Architecture

The framework operates in three distinct phases:

1. **Extraction:** Parses raw agent logs into structured JSON (Planner, Router, and Tool traces).
2. **Scientific Evaluation:**
* **Reasoning-First:** Judges must extract evidence before scoring.
* **Ensemble Scoring:** Multiple iterations with `temperature: 0`.
* **CISC:** Weighted scoring based on the judge's self-assessed confidence.


3. **Observability:** Generates a real-time HTML dashboard to visualize precision, confidence, and judgment variance.

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-repo/agent-eval-framework.git
cd agent-eval-framework

```


2. **Set up the environment:**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



---

## 🚀 Usage Workflow

### 1. Data Preprocessing

Extract structured traces from your raw agent JSON files:

```bash
python preprocess/data_extract.py

```

### 2. Run Scientific Evaluation

Execute the ensemble judge against your ground truth. This uses the OpenRouter API to calculate precision and plan alignment:

```bash
python eval_tools/custom_metric_agent_plan_precision_v2.py

```

### 3. Generate Observability Report

Transform the evaluation results into a high-density dashboard:

```bash
python report\report_gen.py

```

---

## 📊 Observability Dashboard

The framework generates a `C:\Generative_AI_Projects\gen_test_x\report\observability_report.html` containing:

* **Precision vs. Confidence:** A dual-axis line chart to detect "confident hallucinations."
* **Judgment Variance:** A bar chart tracking the standard deviation of scores (signals when a rubric is too vague).
* **Detailed Trace Table:** Every evaluation step with "RE-EVALUATE" flags for high-variance results.

---

## 🧪 Research-Backed Strategies Implemented

* **Ditch the 1–10 Scale:** Uses discrete pointwise rubrics (0, 1, 2) to increase inter-annotator agreement.
* **Position Swapping:** Built-in support for pairwise comparison to eliminate lead-bias.
* **CISC (arXiv:2502.06233):** Implements weighted majority votes where high-confidence judgments carry more weight.

---

## 📋 Requirements

* Python 3.10+
* OpenRouter API Key (for GPT-4o-mini or Claude 3.5 Sonnet)
* Libraries: `openai`, `numpy`, `pathlib`

---
