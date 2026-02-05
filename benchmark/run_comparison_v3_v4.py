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

    # Use first 5 questions for quick benchmark
    questions = questions[:5] 

    results = []
    
    # Scripts
    v3_script = "agents/meta_cognitive_rag_v3.py"
    v4_script = "agents/meta_cognitive_rag_v4.py"

    print(f"🚀 Starting Subprocess Benchmark: V3 (LLM Listwise) vs V4 (BGE Local) on {len(questions)} questions...")
    print("-" * 60)

    for i, q_data in enumerate(questions):
        question = q_data["question"]
        ref_answer = q_data["reference_answer"]
        key_facts = q_data["key_facts"]
        
        print(f"Question {i+1}: {question}")
        
        # --- Run V3 ---
        print("  Running V3 (LLM Listwise)...")
        start_t = time.time()
        v3_out, v3_code = run_agent_subprocess(v3_script, question)
        v3_time = time.time() - start_t
        if v3_code != 0: print(f"  ⚠️ V3 Failed: {v3_out[-200:]}")
        
        # --- Run V4 ---
        print("  Running V4 (BGE Local)...")
        start_t = time.time()
        v4_out, v4_code = run_agent_subprocess(v4_script, question)
        v4_time = time.time() - start_t
        if v4_code != 0: print(f"  ⚠️ V4 Failed: {v4_out[-200:]}")

        # --- Analyze ---
        v3_ans = extract_answer(v3_out)
        v4_ans = extract_answer(v4_out)
        
        v3_rouge = calculate_rouge(ref_answer, v3_ans)
        v4_rouge = calculate_rouge(ref_answer, v4_ans)
        
        v3_recall = check_recall_from_logs(v3_out, key_facts)
        v4_recall = check_recall_from_logs(v4_out, key_facts)
        
        result_row = {
            "id": i+1,
            "question": question,
            "v3_time": v3_time,
            "v4_time": v4_time,
            "v3_rouge": v3_rouge,
            "v4_rouge": v4_rouge,
            "v3_recall": v3_recall,
            "v4_recall": v4_recall,
            "v3_answer": v3_ans[:100].replace("\n", " "),
            "v4_answer": v4_ans[:100].replace("\n", " ")
        }
        results.append(result_row)
        
        print(f"  V3 -> Recall: {v3_recall:.1f}%, ROUGE: {v3_rouge:.1f}, Time: {v3_time:.1f}s")
        print(f"  V4 -> Recall: {v4_recall:.1f}%, ROUGE: {v4_rouge:.1f}, Time: {v4_time:.1f}s")
        
        # Performance check
        time_diff = v3_time - v4_time
        if time_diff < 0:
            print(f"  ⚡ V3 is {abs(time_diff):.1f}s FASTER")
        else:
            print(f"  🐢 V3 is {time_diff:.1f}s SLOWER (BGE wins)")
        
        print("-" * 60)

    # Save Report
    df = pd.DataFrame(results)
    
    # Calculate Averages
    avg_metrics = {
        "v3_avg_recall": df["v3_recall"].mean(),
        "v4_avg_recall": df["v4_recall"].mean(),
        "v3_avg_rouge": df["v3_rouge"].mean(),
        "v4_avg_rouge": df["v4_rouge"].mean(),
        "v3_avg_time": df["v3_time"].mean(),
        "v4_avg_time": df["v4_time"].mean()
    }
    
    print("\n🏆 Benchmark Results Summary 🏆")
    print(f"V3 Avg Time:   {avg_metrics['v3_avg_time']:.2f}s | V4 Avg Time:   {avg_metrics['v4_avg_time']:.2f}s")
    print(f"V3 Avg Recall: {avg_metrics['v3_avg_recall']:.2f}% | V4 Avg Recall: {avg_metrics['v4_avg_recall']:.2f}%")
    print(f"V3 Avg ROUGE:  {avg_metrics['v3_avg_rouge']:.2f}   | V4 Avg ROUGE:  {avg_metrics['v4_avg_rouge']:.2f}")
    
    # Generate Markdown Report
    report = f"""# V3 (LLM) vs V4 (BGE) Benchmark Report

## Summary
| Metric | V3 (LLM Listwise) | V4 (BGE M3 Local) | Improvement (V3 vs V4) |
|--------|-------------------|-------------------|------------------------|
| **Avg Latency** | {avg_metrics['v3_avg_time']:.2f}s | {avg_metrics['v4_avg_time']:.2f}s | {avg_metrics['v4_avg_time'] - avg_metrics['v3_avg_time']:.2f}s |
| **Avg Recall** | {avg_metrics['v3_avg_recall']:.2f}% | {avg_metrics['v4_avg_recall']:.2f}% | {avg_metrics['v3_avg_recall'] - avg_metrics['v4_avg_recall']:.2f}% |
| **Avg ROUGE-L** | {avg_metrics['v3_avg_rouge']:.2f} | {avg_metrics['v4_avg_rouge']:.2f} | {avg_metrics['v3_avg_rouge'] - avg_metrics['v4_avg_rouge']:.2f} |

## Detailed Results
| ID | Question | V3 Time | V4 Time | V3 ROUGE | V4 ROUGE |
|----|----------|---------|---------|----------|----------|
"""
    for index, row in df.iterrows():
        report += f"| {row['id']} | {row['question']} | {row['v3_time']:.1f}s | {row['v4_time']:.1f}s | {row['v3_rouge']:.1f} | {row['v4_rouge']:.1f} |\n"
    
    with open("benchmark_report_v3_v4.md", "w") as f:
        f.write(report)
        
    print("Report saved to benchmark_report_v3_v4.md")

if __name__ == "__main__":
    main()
