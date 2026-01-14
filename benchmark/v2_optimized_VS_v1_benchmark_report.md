# Meta Cognitive RAG V1 vs V2 Benchmark Report

## Summary
| Metric | V1 (Baseline) | V2 (Hybrid + DeepSeek) | Improvement |
|--------|---------------|------------------------|-------------|
| **Avg Recall (Key Facts)** | 60.67% | 62.67% | 2.00% |
| **Avg ROUGE-L (Quality)** | 36.13 | 36.90 | 0.76 |

## Detailed Results
| ID | Question | V1 Recall | V2 Recall | V1 ROUGE | V2 ROUGE | V1 Time | V2 Time |
|----|----------|-----------|-----------|----------|----------|---------|---------|
| 1 | What is the exact battery capacity (in kWh) and the corresponding CLTC range for the StarEra ES9 in its longest-range configuration? | 50.0% | 50.0% | 31.1 | 41.7 | 16.2s | 11.7s |
| 2 | What are the precise length, width, and height dimensions of the StarEra ES9, and what is its wheelbase? | 20.0% | 20.0% | 24.5 | 32.6 | 19.0s | 11.0s |
| 3 | What is the maximum DC fast charging power supported by the StarEra ES9, and how long does it take to charge from 10% to 80%? | 50.0% | 50.0% | 40.0 | 30.0 | 11.0s | 11.9s |
| 4 | What is the 0-100 km/h acceleration time for the dual-motor, all-wheel-drive version of the Xiaomi SU7 Max? | 100.0% | 100.0% | 49.0 | 49.0 | 11.0s | 10.6s |
| 5 | What is the combined peak power output of the dual motors in the top-performance variant of the Xiaomi SU7? | 100.0% | 100.0% | 34.5 | 41.7 | 10.7s | 11.1s |
| 6 | What is the maximum voltage platform of the Xiaomi SU7, and what is its claimed peak charging rate in km of range gained per minute? | 40.0% | 60.0% | 42.6 | 27.7 | 16.2s | 27.5s |
| 7 | Which vehicle, the StarEra ES9 or the Xiaomi SU7, offers a longer maximum CLTC range, and what are the respective figures? | 50.0% | 50.0% | 42.4 | 34.2 | 13.9s | 17.0s |
| 8 | Compare the intelligent driving system chips used in the StarEra ES9 and the Xiaomi SU7. Which specific chips do they use? | 50.0% | 50.0% | 31.6 | 32.3 | 18.2s | 17.7s |
| 9 | Which car has a higher top speed, the StarEra ES9 or the Xiaomi SU7, and what are their respective top speeds? | 80.0% | 80.0% | 53.5 | 63.6 | 19.8s | 18.0s |
| 10 | Compare the passenger seating capacity and configuration of the StarEra ES9 and the Xiaomi SU7. | 66.7% | 66.7% | 12.1 | 16.2 | 22.4s | 18.2s |

## Sample Answers
| ID | V1 Answer | V2 Answer |
|----|-----------|-----------|
| 1 | The StarEra ES9's longest-range configuration, the Performance version, has a 150kWh semi-solid-stat | The StarEra ES9 in its longest-range configuration (Performance version) has a battery capacity of 1 |
| 2 | I am sorry, but I was unable to find the precise length, width, height, and wheelbase dimensions of  | The StarEra ES9 has the following dimensions:  *   **Length:** 4980mm *   **Width:** 1980mm *   **He |
| 3 | The StarEra ES9 supports a maximum DC fast charging power of 350kW [1]. Using a 350kW charging stati | The StarEra ES9 supports a maximum DC fast charging power of 350kW [Page 6, 23]. The charging time f |
| 4 | The 0-100 km/h acceleration time for the dual-motor, all-wheel-drive version of the Xiaomi SU7 Max i | The 0-100 km/h acceleration time for the dual-motor, all-wheel-drive version of the Xiaomi SU7 Max i |
| 5 | The combined peak power output of the dual motors in the top-performance variant (SU7 Max) of the Xi | The combined peak power output of the dual motors in the top-performance variant (SU7 Max) of the Xi |
| 6 | I am sorry, but I am unable to find the maximum voltage platform or the claimed peak charging rate i | The maximum voltage platform of the Xiaomi SU7 is not explicitly stated in the provided documents. H |
| 7 | The Star Era ES9 offers a slightly longer maximum CLTC range compared to the Xiaomi SU7.  *   **Star | The StarEra ES9 offers a longer maximum CLTC range compared to the Xiaomi SU7. The ES9 performance v |
| 8 | The StarEra ES9 uses a self-developed Nebula automatic driving computing chip with a computing power | The StarEra ES9 uses the 星辰自研Nebula自动驾驶计算芯片 (Xingchen Self-Developed Nebula Autonomous Driving Compu |
| 9 | The Xiaomi SU7 Max has a higher top speed at 265 km/h. The Xiaomi SU7 Pro has a top speed of 210 km/ | The Xiaomi SU7 Max has a higher top speed (265 km/h) than the StarEra ES9 (220 km/h). The base model |
| 10 | Here's a comparison of the passenger seating capacity and configuration of the StarEra ES9 and the X | Here's a comparison of the passenger seating capacity and configuration of the StarEra ES9 and the X |
