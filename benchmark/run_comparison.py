
import os
import json
import time
import subprocess
import pandas as pd
from rouge_score import rouge_scorer
import sys

def run_agent_subprocess(script_path, query):
    """Run agent script as subprocess and capture stdout."""
    try:
        cmd = [sys.executable, script_path, query]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            errors='ignore'
        )
        return result.stdout, result.returncode
    except Exception as e:
        return str(e), -1

def extract_answer(stdout):
    """Extract final AI answer from stdout logs."""
    # Look for the last "Ai Message" block or just the last few lines if structured output fails
    # The logs usually show: "================================== Ai Message =================================="
    marker = "================================== Ai Message =================================="
    if marker in stdout:
        return stdout.split(marker)[-1].strip()
    return stdout[-500:] # Fallback

def check_recall_from_logs(stdout, key_facts):
    """Check if key facts appear in the retrieval logs."""
    # We check the whole log because the tool output is printed there.
    # Simple case-insensitive check
    if not key_facts: return 0.0
    
    text = stdout.lower()
    hits = 0
    for fact in key_facts:
        if str(fact).lower() in text:
            hits += 1
    return (hits / len(key_facts)) * 100.0

def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure * 100.0

def main():
    questions_file = "benchmark/questions.json"
    if not os.path.exists(questions_file):
        print("Questions file not found!")
        return

    with open(questions_file, "r") as f:
        questions = json.load(f)

    results = []
    
    # Scripts
    v1_script = "agents/meta_cognitive_rag.py"
    v2_script = "agents/meta_cognitive_rag_v2.py"

    print(f"🚀 Starting Subprocess Benchmark on {len(questions)} questions...")
    print("-" * 60)

    for i, q_data in enumerate(questions):
        question = q_data["question"]
        ref_answer = q_data["reference_answer"]
        key_facts = q_data["key_facts"]
        
        print(f"Question {i+1}: {question}")
        
        # --- Run V1 ---
        start_t = time.time()
        v1_out, v1_code = run_agent_subprocess(v1_script, question)
        v1_time = time.time() - start_t
        if v1_code != 0: print(f"  ⚠️ V1 Failed: {v1_out[-200:]}")
        
        # --- Run V2 ---
        start_t = time.time()
        v2_out, v2_code = run_agent_subprocess(v2_script, question)
        v2_time = time.time() - start_t
        if v2_code != 0: print(f"  ⚠️ V2 Failed: {v2_out[-200:]}")

        # --- Analyze ---
        v1_ans = extract_answer(v1_out)
        v2_ans = extract_answer(v2_out)
        
        v1_rouge = calculate_rouge(ref_answer, v1_ans)
        v2_rouge = calculate_rouge(ref_answer, v2_ans)
        
        v1_recall = check_recall_from_logs(v1_out, key_facts)
        v2_recall = check_recall_from_logs(v2_out, key_facts)
        
        result_row = {
            "id": i+1,
            "question": question,
            "v1_time": v1_time,
            "v2_time": v2_time,
            "v1_rouge": v1_rouge,
            "v2_rouge": v2_rouge,
            "v1_recall": v1_recall,
            "v2_recall": v2_recall,
            "v1_answer": v1_ans[:100].replace("\n", " "),
            "v2_answer": v2_ans[:100].replace("\n", " ")
        }
        results.append(result_row)
        
        print(f"  V1 -> Recall: {v1_recall:.1f}%, ROUGE: {v1_rouge:.1f}, Time: {v1_time:.1f}s")
        print(f"  V2 -> Recall: {v2_recall:.1f}%, ROUGE: {v2_rouge:.1f}, Time: {v2_time:.1f}s")
        print("-" * 60)

    # Save Report
    df = pd.DataFrame(results)
    
    # Calculate Averages
    avg_metrics = {
        "v1_avg_recall": df["v1_recall"].mean(),
        "v2_avg_recall": df["v2_recall"].mean(),
        "v1_avg_rouge": df["v1_rouge"].mean(),
        "v2_avg_rouge": df["v2_rouge"].mean()
    }
    
    print("\n🏆 Benchmark Results Summary 🏆")
    print(f"V1 Avg Recall: {avg_metrics['v1_avg_recall']:.2f}% | V2 Avg Recall: {avg_metrics['v2_avg_recall']:.2f}%")
    print(f"V1 Avg ROUGE:  {avg_metrics['v1_avg_rouge']:.2f}   | V2 Avg ROUGE:  {avg_metrics['v2_avg_rouge']:.2f}")
    
    # Generate Markdown Report
    report = f"""# Meta Cognitive RAG V1 vs V2 Benchmark Report

## Summary
| Metric | V1 (Baseline) | V2 (Hybrid + DeepSeek) | Improvement |
|--------|---------------|------------------------|-------------|
| **Avg Recall (Key Facts)** | {avg_metrics['v1_avg_recall']:.2f}% | {avg_metrics['v2_avg_recall']:.2f}% | {avg_metrics['v2_avg_recall'] - avg_metrics['v1_avg_recall']:.2f}% |
| **Avg ROUGE-L (Quality)** | {avg_metrics['v1_avg_rouge']:.2f} | {avg_metrics['v2_avg_rouge']:.2f} | {avg_metrics['v2_avg_rouge'] - avg_metrics['v1_avg_rouge']:.2f} |

## Detailed Results
| ID | Question | V1 Recall | V2 Recall | V1 ROUGE | V2 ROUGE | V1 Time | V2 Time |
|----|----------|-----------|-----------|----------|----------|---------|---------|
"""
    for index, row in df.iterrows():
        report += f"| {row['id']} | {row['question']} | {row['v1_recall']:.1f}% | {row['v2_recall']:.1f}% | {row['v1_rouge']:.1f} | {row['v2_rouge']:.1f} | {row['v1_time']:.1f}s | {row['v2_time']:.1f}s |\n"
    
    # Append Answers Table
    report += "\n## Sample Answers\n| ID | V1 Answer | V2 Answer |\n|----|-----------|-----------|\n"
    for index, row in df.iterrows():
         report += f"| {row['id']} | {row['v1_answer']} | {row['v2_answer']} |\n"

    with open("benchmark_report.md", "w") as f:
        f.write(report)
        
    print("Report saved to benchmark_report.md")

if __name__ == "__main__":
    main()
