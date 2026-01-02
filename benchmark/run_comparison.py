
import subprocess
import time
import re
from rouge_score import rouge_scorer
import numpy as np
import os

# --- Configuration ---
TEST_SCRIPT = "deep_agent_comparison.py"
TEMP_SCRIPT_NO_THINK = "deep_agent_baseline.py"

# Ground Truth (Updated with Table Data)
ground_truth = """
星辰ES9与小米SU7的续航表现都非常出色。
小米SU7的官方CLTC续航数据为：标准版700km，Pro版830km，Max版800km/810km。
星辰ES9通过新提取的表格数据显示，虽然官方强调实际达成率（85%-90%），但其理论续航范围也在600km至850km之间（视具体版本而定）。
综合来看，两者的顶级版本续航能力在同一梯队（800km+），但星辰ES9更强调不同工况下的真实保有率。
"""

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def run_agent_test(script_path, description):
    print(f"\n--- Running Test: {description} ---")
    start_time = time.time()
    
    try:
        # Run the script with a timeout (agents can get stuck)
        result = subprocess.run(
            [".venv/bin/python3", script_path], 
            capture_output=True, 
            text=True, 
            timeout=120
        )
        
        output = result.stdout
        stderr = result.stderr
        
        if result.returncode != 0:
            print(f"Error running script: {stderr}")
            # Even if error, output might have debug info
            if not output: output = stderr

        # --- Extract Metrics ---
        
        # 1. Final Answer Extraction
        final_answer = ""
        # Look for different possible markers
        if "=== FINAL ANSWER ===" in output:
             final_answer = output.split("=== FINAL ANSWER ===")[-1].strip()
        elif "Ai Message" in output:
             # Capture the LAST Ai Message content
             parts = output.split("================================== Ai Message ==================================")
             if parts:
                 final_answer = parts[-1].split("--- Update")[0].strip()
        
        if not final_answer or len(final_answer) < 10:
             # Fallback: Get the last 1000 chars
             final_answer = "[RAW TAIL]\n" + output[-1000:]

        # 2. Step Count
        steps_count = output.count("--- Update from node: model ---")
        
        # 3. Reflection Count
        reflection_count = output.count("reflection:") 
        
        duration = time.time() - start_time
        
        # 4. ROUGE Score
        scores = scorer.score(ground_truth, final_answer)
        rouge_l = scores['rougeL'].fmeasure
        
        # 5. Length
        response_len = len(final_answer)
        
        # 6. Extract DEBUG logs from our hybrid_rag_tool
        debug_logs = "\n".join([line for line in output.split('\n') if "DEBUG:" in line])

        print(f"  Duration: {duration:.2f}s")
        print(f"  Steps: {steps_count}")
        print(f"  Reflections: {reflection_count}")
        print(f"  ROUGE-L: {rouge_l:.4f}")
        print(f"  Response Length: {response_len} chars")
        
        return {
            "duration": duration,
            "steps": steps_count,
            "reflections": reflection_count,
            "rouge_l": rouge_l,
            "length": response_len,
            "answer": final_answer,
            "debug_logs": debug_logs
        }
        
    except subprocess.TimeoutExpired:
        print("Error: Test timed out.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("preparing tests...")
    
    # Run Comparisons
    results_base = run_agent_test(TEMP_SCRIPT_NO_THINK, "Baseline (No Think Tool)")
    results_meta = run_agent_test(TEST_SCRIPT, "Meta-Cognitive (With Think Tool)")
    
    if not results_base or not results_meta:
        print("Tests failed.")
        return

    # Print Report
    print("\n\n" + "="*60)
    print(f"{'METRIC':<20} | {'BASELINE':<15} | {'META-COGNITIVE':<15} | {'IMPACT':<10}")
    print("-" * 75)
    
    metrics = [
        ("Response Quality (ROUGE-L)", "rouge_l", True),
        ("Information Density (Len)", "length", False),
        ("Reasoning Steps", "steps", True),
        ("Self-Reflections", "reflections", True)
    ]
    
    for label, key, higher_better in metrics:
        v1 = results_base.get(key, 0)
        v2 = results_meta.get(key, 0)
        
        if v1 == 0:
            change = 0
        else:
            change = ((v2 - v1) / v1) * 100
            
        sign = "+" if change > 0 else ""
        
        print(f"{label:<20} | {v1:<15.4f} | {v2:<15.4f} | {sign}{change:.1f}%")
    
    print("\n" + "="*60)
    print("generated answers comparison")
    print("="*60)
    print(f"\n[baseline (no think tool)]:\n{results_base.get('answer', 'N/A')}")
    print("\n[Retrieval Logs]:")
    print(results_base.get('debug_logs', 'No logs'))
    
    print("\n" + "-"*30)
    
    print(f"\n[meta-cognitive (with think tool)]:\n{results_meta.get('answer', 'N/A')}")
    print("\n[Retrieval Logs]:")
    print(results_meta.get('debug_logs', 'No logs'))
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
