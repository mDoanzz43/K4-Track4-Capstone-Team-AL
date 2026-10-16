# Evaluation report

Generated: `2026-10-16T10:04:40.622873+00:00`  
Commit: `6403348799f067ebb9e3adda61dc55c68911c396`  
Dataset: synthetic, no personal/patient/student data.

| Metric | Result |
|---|---:|
| Vision cases passed | 9/9 |
| Mean / minimum ink coverage | 99.994% / 99.946% |
| Mean vision latency | 49.19 ms/image |
| Mean robot tracking RMS | 0.415 mm |
| Worst robot maximum error | 0.438 mm |
| Known failures reproduced | 1/1 |
| Validation cases passed | 1/1 |

These results describe this synthetic dataset and PyBullet simulation only. They
are not claims about OCR accuracy, real robot accuracy, or webcam gesture accuracy.
See `metrics.json` for per-case measurements and `failed_cases.md` for failure analysis.
