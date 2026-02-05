# V2 vs V3 (Optimized) Benchmark Report

## Summary
| Metric | V2 (Pointwise Rerank) | V3 (Listwise Batched Rerank) | Improvement |
|--------|-----------------------|-----------------------------|-------------|
| **Avg Latency (Speed)** | 17.31s | 16.47s | 0.84s (4.8%) |
| **Avg Recall (Key Facts)** | 50.33% | 57.33% | 7.00% |
| **Avg ROUGE-L (Quality)** | 22.85 | 31.22 | 8.37 |

## Detailed Results
| ID | Question | V2 Time | V3 Time | V2 Recall | V3 Recall | V2 ROUGE | V3 ROUGE |
|----|----------|---------|---------|-----------|-----------|----------|----------|
| 1 | What is the exact battery capacity (in kWh) and the corresponding CLTC range for the StarEra ES9 in its longest-range configuration? | 18.5s | 13.1s | 50.0% | 50.0% | 15.4 | 41.7 |
| 2 | What are the precise length, width, and height dimensions of the StarEra ES9, and what is its wheelbase? | 11.7s | 10.6s | 20.0% | 20.0% | 26.7 | 29.8 |
| 3 | What is the maximum DC fast charging power supported by the StarEra ES9, and how long does it take to charge from 10% to 80%? | 15.6s | 10.6s | 50.0% | 50.0% | 10.8 | 22.4 |
| 4 | What is the 0-100 km/h acceleration time for the dual-motor, all-wheel-drive version of the Xiaomi SU7 Max? | 17.2s | 9.7s | 75.0% | 100.0% | 36.7 | 55.8 |
| 5 | What is the combined peak power output of the dual motors in the top-performance variant of the Xiaomi SU7? | 18.7s | 11.1s | 75.0% | 100.0% | 25.0 | 51.1 |
| 6 | What is the maximum voltage platform of the Xiaomi SU7, and what is its claimed peak charging rate in km of range gained per minute? | 16.3s | 30.2s | 40.0% | 60.0% | 35.1 | 17.8 |
| 7 | Which vehicle, the StarEra ES9 or the Xiaomi SU7, offers a longer maximum CLTC range, and what are the respective figures? | 14.0s | 18.2s | 50.0% | 50.0% | 9.8 | 34.7 |
| 8 | Compare the intelligent driving system chips used in the StarEra ES9 and the Xiaomi SU7. Which specific chips do they use? | 20.4s | 26.4s | 50.0% | 50.0% | 29.6 | 17.1 |
| 9 | Which car has a higher top speed, the StarEra ES9 or the Xiaomi SU7, and what are their respective top speeds? | 18.8s | 17.9s | 60.0% | 60.0% | 21.1 | 21.8 |
| 10 | Compare the passenger seating capacity and configuration of the StarEra ES9 and the Xiaomi SU7. | 21.9s | 16.9s | 33.3% | 33.3% | 18.5 | 20.0 |
