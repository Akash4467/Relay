## Latest measurement (same 25 questions: 20 answerable, 5 not; real Groq, setting B)

| Configuration | Correct | Partly | Wrong | Refused | Unanswerable refused | Cites retrieved | p50 s | Requests / tokens |
|---|---|---|---|---|---|---|---|---|
| Production (c3, p4, v4.2) | 2/20 | 1/20 | 1/20 | 16/20 | 5/5 | 5/5 | 28.6 | several runs |
| Candidate v4.7c (c6, R3b, p7) | 8/20 | 4/20 | 2/20 | 6/20 | 5/5 | 22/22 | 34.5 | 41 / 93,002 |
| Candidate v4.8c (c6, R3b, p7) | 9/20 | 5/20 | 0/20 | 6/20 | 5/5 | 25/25 | 33.4 | 41 / 93,658 |

Live rows: one run per question, AI-graded; no wrong answer in one run is not a zero error rate. A small live check of v4.8c came first (7 questions: 5 correct, 2 refused). Offline, its two safeguards changed only the 2 target drafts of 401 replayed cases and did not have to act live; 541 tests pass. Source: assessment/e2e/.
