import json
import os

def generate_html_report(eval_results_path, output_path="observability_report.html"):
    # 1. Load your evaluation data
    with open(eval_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Extract Data for Charts
    # Handles both 'precision_eval' and 'ensemble_eval' formats
    details = data.get("details", [])
    labels = [f"Step {d.get('step', i+1)}" for i, d in enumerate(details)]
    precision_data = [d.get("precision", d.get("weighted_avg", 0)) for d in details]
    confidence_data = [d.get("confidence", sum([j.get('confidence', 0) for j in d.get('details', [])])/len(d.get('details', [1]))) for d in details]
    variance_data = [d.get("variance_std", 0) for d in details]

    # 3. Generate Table Rows
    table_rows = ""
    for d in details:
        status_class = "status-healthy" if d.get("variance_std", 0) < 0.4 else "status-critical"
        status_text = "VERIFIED" if status_class == "status-healthy" else "RE-EVALUATE"
        
        table_rows += f"""
        <tr>
            <td>Step_{d.get('step', 'N/A')}</td>
            <td>{d.get('precision', d.get('final_score', 0))}</td>
            <td>{d.get('confidence', 'N/A')}</td>
            <td>{d.get('variance_std', 0)}</td>
            <td><span class="status-badge {status_class}">{status_text}</span></td>
        </tr>
        """

    # 4. The HTML Template with Injected Data
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Generative AI Techshila Agent Evaluation Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: 'Inter', sans-serif; background-color: #0d1117; color: #c9d1d9; padding: 20px; }}
            .dashboard-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
            .metric-value {{ font-size: 2rem; font-weight: bold; color: #58a6ff; }}
            .status-badge {{ padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }}
            .status-critical {{ background: #da3633; color: white; }}
            .status-healthy {{ background: #238636; color: white; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #30363d; }}
        </style>
    </head>
    <body>
        <h1>🚀 Generative AI Techshila </h1>
         <h1> Agent Observability Report</h1>
        
        <div class="dashboard-grid">
            <div class="card">
                <div>AVG PRECISION</div>
                <div class="metric-value">{data.get('average_precision', data.get('avg_precision', '0.00'))}</div>
            </div>
            <div class="card">
                <div>AVG CONFIDENCE</div>
                <div class="metric-value">{data.get('avg_confidence', 'N/A')}</div>
            </div>
        </div>

        <div class="dashboard-grid">
            <div class="card">
                <canvas id="precisionChart"></canvas>
            </div>
            <div class="card">
                <canvas id="varianceChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h3>Detailed Trace</h3>
            <table>
                <thead><tr><th>ID</th><th>Score</th><th>Conf</th><th>Var</th><th>Status</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>

        <script>
            new Chart(document.getElementById('precisionChart'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: [
                        {{ label: 'Precision', data: {json.dumps(precision_data)}, borderColor: '#58a6ff' }},
                        {{ label: 'Confidence', data: {json.dumps(confidence_data)}, borderColor: '#3fb950', borderDash: [5,5] }}
                    ]
                }}
            }});
            new Chart(document.getElementById('varianceChart'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: [{{ label: 'Variance', data: {json.dumps(variance_data)}, backgroundColor: '#da3633' }}]
                }}
            }});
        </script>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Dashboard generated: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    # Point this to your result JSON file
    generate_html_report(r"scientific_eval_report.json")