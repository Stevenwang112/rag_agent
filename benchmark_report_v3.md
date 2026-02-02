# V2 vs V3 (Optimized) Benchmark Report

## Summary
| Metric | V2 (Pointwise Rerank) | V3 (Listwise Batched Rerank) | Improvement |
|--------|-----------------------|-----------------------------|-------------|
| **Avg Latency (Speed)** | 2.72s | 6.73s | -4.01s (-147.8%) |
| **Avg Recall (Key Facts)** | 0.00% | 5.00% | 5.00% |
| **Avg ROUGE-L (Quality)** | 0.00 | 2.78 | 2.78 |

## Detailed Results
| ID | Question | V2 Time | V3 Time | V2 Recall | V3 Recall | V2 ROUGE | V3 ROUGE |
|----|----------|---------|---------|-----------|-----------|----------|----------|
| 1 | What is the exact battery capacity (in kWh) and the corresponding CLTC range for the StarEra ES9 in its longest-range configuration? | 6.2s | 4.6s | 0.0% | 0.0% | 0.0 | 0.0 |
| 2 | What are the precise length, width, and height dimensions of the StarEra ES9, and what is its wheelbase? | 2.4s | 4.5s | 0.0% | 0.0% | 0.0 | 0.0 |
| 3 | What is the maximum DC fast charging power supported by the StarEra ES9, and how long does it take to charge from 10% to 80%? | 2.4s | 4.5s | 0.0% | 0.0% | 0.0 | 0.0 |
| 4 | What is the 0-100 km/h acceleration time for the dual-motor, all-wheel-drive version of the Xiaomi SU7 Max? | 2.3s | 4.5s | 0.0% | 0.0% | 0.0 | 0.0 |
| 5 | What is the combined peak power output of the dual motors in the top-performance variant of the Xiaomi SU7? | 2.3s | 4.5s | 0.0% | 0.0% | 0.0 | 0.0 |
| 6 | What is the maximum voltage platform of the Xiaomi SU7, and what is its claimed peak charging rate in km of range gained per minute? | 2.3s | 4.5s | 0.0% | 0.0% | 0.0 | 0.0 |
| 7 | Which vehicle, the StarEra ES9 or the Xiaomi SU7, offers a longer maximum CLTC range, and what are the respective figures? | 2.3s | 4.5s | 0.0% | 0.0% | 0.0 | 0.0 |
| 8 | Compare the intelligent driving system chips used in the StarEra ES9 and the Xiaomi SU7. Which specific chips do they use? | 2.4s | 4.6s | 0.0% | 0.0% | 0.0 | 0.0 |
| 9 | Which car has a higher top speed, the StarEra ES9 or the Xiaomi SU7, and what are their respective top speeds? | 2.3s | 4.5s | 0.0% | 0.0% | 0.0 | 0.0 |
| 10 | Compare the passenger seating capacity and configuration of the StarEra ES9 and the Xiaomi SU7. | 2.3s | 26.7s | 0.0% | 50.0% | 0.0 | 27.8 |
