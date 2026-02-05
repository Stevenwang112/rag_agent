# V3 (LLM) vs V4 (BGE) Benchmark Report

## Summary
| Metric | V3 (LLM Listwise) | V4 (BGE M3 Local) | Improvement (V3 vs V4) |
|--------|-------------------|-------------------|------------------------|
| **Avg Latency** | 12.84s | 14.11s | 1.27s |
| **Avg Recall** | 54.00% | 54.00% | 0.00% |
| **Avg ROUGE-L** | 30.24 | 26.56 | 3.68 |

## Detailed Results
| ID | Question | V3 Time | V4 Time | V3 ROUGE | V4 ROUGE |
|----|----------|---------|---------|----------|----------|
| 1 | What is the exact battery capacity (in kWh) and the corresponding CLTC range for the StarEra ES9 in its longest-range configuration? | 11.1s | 15.6s | 16.0 | 17.8 |
| 2 | What are the precise length, width, and height dimensions of the StarEra ES9, and what is its wheelbase? | 22.3s | 15.3s | 18.6 | 17.8 |
| 3 | What is the maximum DC fast charging power supported by the StarEra ES9, and how long does it take to charge from 10% to 80%? | 9.3s | 13.2s | 38.3 | 26.3 |
| 4 | What is the 0-100 km/h acceleration time for the dual-motor, all-wheel-drive version of the Xiaomi SU7 Max? | 11.8s | 13.0s | 40.8 | 40.7 |
| 5 | What is the combined peak power output of the dual motors in the top-performance variant of the Xiaomi SU7? | 9.7s | 13.5s | 37.5 | 30.2 |
