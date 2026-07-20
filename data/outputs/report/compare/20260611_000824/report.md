# Eval Comparison (means only) — 20260611_000824

E2E_Latency.mean = mean of per-case `avg_ttft_ms`（mean-of-means）；每个 case 的 `avg_ttft_ms` 已是 turn-level 均值（sum/n_assistant），展示层统一标签为 E2E_Latency。

## ① 单模型多轮均值（汇总）

| Label | n_runs | n_cases | ECR | TS | IFS | IISR | AR | Eff | SES | CEI | E2E_Latency(ms) | InputTokens | OutputTokens | TotalTokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.6-plus | 1 | 10 | 0.8500 | 0.3873 | 0.7375 | 0.6682 | 0.6062 | 0.5425 (n=10) | 0.2895 | 16.2507 | 83619.42 | 16108.4 | 11656.0 | 27764.4 |
| gpt-5.3-chat-0303-global | 1 | 10 | 0.9667 | 0.4083 | 0.6000 | 0.6854 | 0.6701 | 0.6084 (n=10) | 0.3873 | 59.0171 | 24888.04 | 9624.2 | 1173.0 | 10797.2 |

### 样本状态统计（按 model 合并）

| Model | total | success | skipped | error | unparseable |
| --- | --- | --- | --- | --- | --- |
| qwen3.6-plus | 10 | 10 | 0 | 0 | 0 |
| gpt-5.3-chat-0303-global | 10 | 10 | 0 | 0 | 0 |

## ② 单模型每轮评测明细

### qwen3.6-plus（1 runs）

| Label | n_runs | n_cases | ECR | TS | IFS | IISR | AR | Eff | SES | CEI | E2E_Latency(ms) | InputTokens | OutputTokens | TotalTokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| evaluation_result_20260611_000224_qwen3.6-plus | 1 | 10 | 0.8500 | 0.3873 | 0.7375 | 0.6682 | 0.6062 | 0.5425 (n=10) | 0.2895 | 16.2507 | 83619.42 | 16108.4 | 11656.0 | 27764.4 |

| Run | total | success | skipped | error | unparseable |
| --- | --- | --- | --- | --- | --- |
| evaluation_result_20260611_000224_qwen3.6-plus | 10 | 10 | 0 | 0 | 0 |

### gpt-5.3-chat-0303-global（1 runs）

| Label | n_runs | n_cases | ECR | TS | IFS | IISR | AR | Eff | SES | CEI | E2E_Latency(ms) | InputTokens | OutputTokens | TotalTokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| evaluation_result_20260611_000704_gpt-5.3-chat-0303-global | 1 | 10 | 0.9667 | 0.4083 | 0.6000 | 0.6854 | 0.6701 | 0.6084 (n=10) | 0.3873 | 59.0171 | 24888.04 | 9624.2 | 1173.0 | 10797.2 |

| Run | total | success | skipped | error | unparseable |
| --- | --- | --- | --- | --- | --- |
| evaluation_result_20260611_000704_gpt-5.3-chat-0303-global | 10 | 10 | 0 | 0 | 0 |
