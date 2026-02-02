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
    marker = "================================== Ai Message =================================="
    if marker in stdout:
        return stdout.split(marker)[-1].strip()
    return stdout[-500:] # Fallback

def check_recall_from_logs(stdout, key_facts):
    """Check if key facts appear in the retrieval logs."""
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

    # Use first 5 questions for quick benchmark if list is long
    questions = questions[:] 

    results = []
    
    # Scripts
    v2_script = "agents/meta_cognitive_rag_v2.py"
    v3_script = "agents/meta_cognitive_rag_v3.py"

    print(f"🚀 Starting Subprocess Benchmark: V2 vs V3 (Optimized) on {len(questions)} questions...")
    print("-" * 60)

    for i, q_data in enumerate(questions):
        question = q_data["question"]
        ref_answer = q_data["reference_answer"]
        key_facts = q_data["key_facts"]
        
        print(f"Question {i+1}: {question}")
        
        # --- Run V2 ---
        print("  Running V2...")
        start_t = time.time()
        v2_out, v2_code = run_agent_subprocess(v2_script, question)
        v2_time = time.time() - start_t
        if v2_code != 0: print(f"  ⚠️ V2 Failed: {v2_out[-200:]}")
        
        # --- Run V3 ---
        print("  Running V3...")
        start_t = time.time()
        v3_out, v3_code = run_agent_subprocess(v3_script, question)
        v3_time = time.time() - start_t
        if v3_code != 0: print(f"  ⚠️ V3 Failed: {v3_out[-200:]}")

        # --- Analyze ---
        v2_ans = extract_answer(v2_out)
        v3_ans = extract_answer(v3_out)
        
        v2_rouge = calculate_rouge(ref_answer, v2_ans)
        v3_rouge = calculate_rouge(ref_answer, v3_ans)
        
        v2_recall = check_recall_from_logs(v2_out, key_facts)
        v3_recall = check_recall_from_logs(v3_out, key_facts)
        
        result_row = {
            "id": i+1,
            "question": question,
            "v2_time": v2_time,
            "v3_time": v3_time,
            "v2_rouge": v2_rouge,
            "v3_rouge": v3_rouge,
            "v2_recall": v2_recall,
            "v3_recall": v3_recall,
            "v2_answer": v2_ans[:100].replace("\n", " "),
            "v3_answer": v3_ans[:100].replace("\n", " ")
        }
        results.append(result_row)
        
        print(f"  V2 -> Recall: {v2_recall:.1f}%, ROUGE: {v2_rouge:.1f}, Time: {v2_time:.1f}s")
        print(f"  V3 -> Recall: {v3_recall:.1f}%, ROUGE: {v3_rouge:.1f}, Time: {v3_time:.1f}s")
        
        # Performance check
        time_diff = v2_time - v3_time
        if time_diff > 0:
            print(f"  ⚡ V3 is {time_diff:.1f}s FASTER")
        else:
            print(f"  🐢 V3 is {abs(time_diff):.1f}s SLOWER")
        
        print("-" * 60)

    # Save Report
    df = pd.DataFrame(results)
    
    # Calculate Averages
    avg_metrics = {
        "v2_avg_recall": df["v2_recall"].mean(),
        "v3_avg_recall": df["v3_recall"].mean(),
        "v2_avg_rouge": df["v2_rouge"].mean(),
        "v3_avg_rouge": df["v3_rouge"].mean(),
        "v2_avg_time": df["v2_time"].mean(),
        "v3_avg_time": df["v3_time"].mean()
    }
    
    print("\n🏆 Benchmark Results Summary 🏆")
    print(f"V2 Avg Time:   {avg_metrics['v2_avg_time']:.2f}s | V3 Avg Time:   {avg_metrics['v3_avg_time']:.2f}s")
    print(f"V2 Avg Recall: {avg_metrics['v2_avg_recall']:.2f}% | V3 Avg Recall: {avg_metrics['v3_avg_recall']:.2f}%")
    print(f"V2 Avg ROUGE:  {avg_metrics['v2_avg_rouge']:.2f}   | V3 Avg ROUGE:  {avg_metrics['v3_avg_rouge']:.2f}")
    
    # Generate Markdown Report
    report = f"""# V2 vs V3 (Optimized) Benchmark Report

## Summary
| Metric | V2 (Pointwise Rerank) | V3 (Listwise Batched Rerank) | Improvement |
|--------|-----------------------|-----------------------------|-------------|
| **Avg Latency (Speed)** | {avg_metrics['v2_avg_time']:.2f}s | {avg_metrics['v3_avg_time']:.2f}s | {avg_metrics['v2_avg_time'] - avg_metrics['v3_avg_time']:.2f}s ({(avg_metrics['v2_avg_time'] - avg_metrics['v3_avg_time'])/avg_metrics['v2_avg_time']*100:.1f}%) |
| **Avg Recall (Key Facts)** | {avg_metrics['v2_avg_recall']:.2f}% | {avg_metrics['v3_avg_recall']:.2f}% | {avg_metrics['v3_avg_recall'] - avg_metrics['v2_avg_recall']:.2f}% |
| **Avg ROUGE-L (Quality)** | {avg_metrics['v2_avg_rouge']:.2f} | {avg_metrics['v3_avg_rouge']:.2f} | {avg_metrics['v3_avg_rouge'] - avg_metrics['v2_avg_rouge']:.2f} |

## Detailed Results
| ID | Question | V2 Time | V3 Time | V2 Recall | V3 Recall | V2 ROUGE | V3 ROUGE |
|----|----------|---------|---------|-----------|-----------|----------|----------|
"""
    for index, row in df.iterrows():
        report += f"| {row['id']} | {row['question']} | {row['v2_time']:.1f}s | {row['v3_time']:.1f}s | {row['v2_recall']:.1f}% | {row['v3_recall']:.1f}% | {row['v2_rouge']:.1f} | {row['v3_rouge']:.1f} |\n"
    
    with open("benchmark_report_v3.md", "w") as f:
        f.write(report)
        
    print("Report saved to benchmark_report_v3.md")

if __name__ == "__main__":
    main()
