# MapSatisfyBench

基于 LangGraph + FastAPI 的 Agent 评测服务。围绕三个核心能力：

1. **被测 Agent 仿真** — 用 LangGraph ReAct 图包装目标 Agent，可挂真实工具或走沙箱 mock
2. **多轮对话仿真** — LLM 模拟用户与 Agent 多轮交互，自动产出对话轨迹
3. **离线判分** — JudgeAgent 出 verdict，8 个确定性指标（ECR / TS / IFS / IISR / AR / Eff / SES / CEI）量化评估

## 目录结构

```
amap-eval-service/
├── app/
│   ├── main.py                          # FastAPI 入口（uvicorn, 4 worker）
│   ├── config.py                        # 配置加载（.env → 默认值）
│   ├── api/routes/
│   │   ├── evaluate.py                  # POST /api/v1/evaluate/case
│   │   ├── simulator.py                 # POST /api/v1/simulate/dialogue
│   │   ├── pipeline.py                  # POST /api/v1/pipeline/run + GET task
│   │   └── health.py                    # /status.taobao, /hello-world
│   ├── cli/
│   │   ├── dialogue_simulator.py        # 多轮对话仿真 CLI
│   │   ├── evaluate.py                  # 单轮批量评测 CLI
│   │   └── test_tools.py               # 工具调试 CLI
│   ├── core/
│   │   ├── simulator/
│   │   │   ├── agent_simulator.py       # LangGraph ReAct Agent 仿真器
│   │   │   ├── dialogue_simulator.py    # 多轮对话循环引擎
│   │   │   ├── user_simulator.py        # LLM 驱动的用户模拟器
│   │   │   └── tool_simulator.py        # 沙箱未命中时的 LLM 工具仿造
│   │   ├── evaluation/
│   │   │   ├── judge_agent.py           # JudgeAgent 五段式评分
│   │   │   ├── schema.py               # GroundTruth / JudgeVerdict / MetricScores
│   │   │   ├── metrics/                 # 9 个确定性指标计算（纯函数）
│   │   │   ├── meta_judge.py            # Meta-Judge 审计（可选）
│   │   │   └── verifiers/               # Web 事实校验（可选）
│   │   └── tools/
│   │       ├── manager.py               # 22 个工具注册清单
│   │       ├── base.py                  # sandbox_cache / safe_tool 装饰器
│   │       └── *.py                     # 各业务工具定义
│   ├── scripts/
│   │   ├── run_pipeline.py              # 端到端流水线（仿真→评分→报告）
│   │   ├── batch_evaluate_from_simulator.py  # 批量评分
│   │   └── compare_eval_results.py      # 多模型评测对比
│   └── services/
│       ├── run_sandbox.py               # 沙箱 mock 匹配（CSV 精确 + 向量模糊）
│       ├── dialogue_recorder.py         # 对话结果写入 CSV
│       ├── eval_summary_service.py      # 评测报告生成
│       └── eval_compare_service.py      # 多模型对比报告
├── data/
│   ├── inputs/                          # 评测用例 xlsx
│   ├── outputs/
│   │   ├── simulator_res/               # 仿真结果 CSV
│   │   ├── evaluation_res/              # 评分结果 CSV
│   │   └── report/                      # 评测报告（HTML/TXT/JSON/PNG）
│   └── sandbox/
│       └── mock_data/                   # 工具沙箱数据（按工具名分 CSV）
├── scripts/                             # 独立辅助脚本
├── tests/                               # 测试
└── pyproject.toml
```

## 环境要求

- Python >= 3.12
- 包管理：**uv**

```bash
uv sync                           # 安装依赖
cp .env.example .env               # 配置 API Key（见下文）
```

### 关键环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `AI_STUDIO_TOKEN` | LLM API Token | 必填 |
| `BASE_URL` | LLM API Base URL | 必填 |
| `MODEL_NAME` | 被测 Agent 默认模型 | `gpt-5.3-chat-0303-global` |

## 快速开始

### 1. 启动 HTTP 服务

```bash
uv run python -m app.main                # 4 worker（默认）
uv run uvicorn app.main:app --port 8080 --reload   # 单 worker 调试
```

### 2. 沙箱模式试跑（不需要外部 API）

```bash
# 多轮对话仿真（沙箱模式，工具走 mock 数据）
.venv/bin/python -m app.cli.dialogue_simulator run demo_bench.xlsx --sandbox

# 端到端流水线（仿真 → 评分 → 报告）
.venv/bin/python -m app.scripts.run_pipeline demo_bench.xlsx --sandbox
```

## CLI 命令

### 多轮对话仿真

```bash
.venv/bin/python -m app.cli.dialogue_simulator run <filename.xlsx> [options]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `filename` | `data/inputs/` 下的 xlsx 文件名 | 必填 |
| `--sandbox` | 沙箱模式（工具走 mock） | `false` |
| `--model MODEL` | 被测 Agent 模型名 | `settings.MODEL_NAME` |
| `--concurrency N` | 并发 case 数 | `4` |
| `--max-turns N` | 最大对话轮次 | `20` |
| `--streaming` | 流式 LLM 调用 | `false` |
| `--thinking` | 推理模式（qwen3/deepseek-v4/gemini） | `false` |
| `--suffix TEXT` | 输出文件名后缀 | 自动用 model 名 |

输出：`data/outputs/simulator_res/dialogue_<filename>_<timestamp>_<suffix>.csv`

### 端到端流水线

```bash
.venv/bin/python -m app.scripts.run_pipeline <filename.xlsx> [options]
```

三个阶段自动串联：

```
Stage 1  仿真  → data/outputs/simulator_res/dialogue_*.csv
Stage 2  评分  → data/outputs/evaluation_res/evaluation_result_*.csv
Stage 3  报告  → data/outputs/report/single/<ts>/summary.*
```

**单模型：**
```bash
.venv/bin/python -m app.scripts.run_pipeline demo_bench.xlsx \
  --sandbox --model qwen3-plus --concurrency 5
```

**多模型对比：**
```bash
.venv/bin/python -m app.scripts.run_pipeline demo_bench.xlsx \
  --sandbox --models "qwen3-plus,gpt-4o,claude-sonnet-4-6" --concurrency 5
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 单模型名 | `settings.MODEL_NAME` |
| `--models` | 多模型对比（逗号分隔，与 `--model` 互斥） | - |
| `--sandbox` | 沙箱模式 | `false` |
| `--concurrency N` | 仿真并发数 | `4` |
| `--eval-concurrency N` | 评分并发数 | `2` |
| `--max-turns N` | 最大对话轮次 | `20` |
| `--streaming` | 流式 LLM | `false` |
| `--thinking` | 推理模式 | `false` |
| `--language` | 评分语言 (`chinese`/`english`) | `chinese` |
| `--enable-verification` | 启用 Web 事实校验 | `false` |
| `--enable-meta-judge` | 启用 Meta-Judge 审计 | `false` |
| `--skip-simulate` | 跳过仿真（搭配 `--sim-result`） | - |
| `--skip-evaluate` | 跳过评分（搭配 `--eval-result`） | - |

### 批量评分（独立使用）

```bash
.venv/bin/python -m app.scripts.batch_evaluate_from_simulator \
  --input data/outputs/simulator_res/<dialogue_result>.csv \
  --cases-input data/inputs/demo_bench.xlsx \
  --max-concurrency 4
```

## HTTP API

### POST /api/v1/evaluate/case

单 case JudgeAgent 评分。

**请求体：**
```json
{
  "case_id": "case-001",
  "query": "帮我找附近的餐厅",
  "full_intent": "找评分4.5以上的中餐厅",
  "conversation_history_messages": [
    {"role": "user", "content": "帮我找附近的餐厅"},
    {"role": "assistant", "content": "..."}
  ],
  "ground_truth": { "explicit_intent": [...], "implicit_intent": [...], ... },
  "language": "chinese",
  "enable_verification": true,
  "enable_meta_judge": false
}
```

**响应体：**
```json
{
  "case_id": "case-001",
  "results": {
    "metrics": { "ECR": 1.0, "TS": 0.8, "IFS": 1.0, "IISR": 0.75, "AR": 0.75, "Eff": 0.55 },
    "details": { "AR": {...}, "IFS": {...}, ... }
  },
  "reason": "..."
}
```

### POST /api/v1/simulate/dialogue

单 case 多轮对话仿真。

**请求体：**
```json
{
  "query": "帮我规划北京三日游",
  "full_intent": "包含故宫、长城、颐和园",
  "current_time": "2026-06-10 14:00",
  "current_location": "北京市朝阳区",
  "model": "qwen3-plus",
  "max_turns": 20,
  "sandbox": true
}
```

**响应体：**
```json
{
  "conversation_id": "uuid",
  "turns": [
    { "turn_index": 0, "role": "assistant", "content": "...", "tool_calls": [...] },
    { "turn_index": 1, "role": "user", "content": "..." }
  ],
  "total_turns": 8,
  "is_natural_stop": true
}
```

### POST /api/v1/pipeline/run

异步流水线（上传 xlsx，后台执行仿真→评分→报告）。

```bash
curl -X POST http://localhost:8080/api/v1/pipeline/run \
  -F "file=@data/inputs/demo_bench.xlsx" \
  -F "sandbox=true" \
  -F "model=qwen3-plus"
```

返回 `eval_task_id`，通过 `GET /api/v1/pipeline/task/{id}` 查询进度或下载产物。

### GET /api/v1/pipeline/task/{eval_task_id}

查询流水线状态或下载产物。

| 参数 | 说明 |
|------|------|
| `?download=input` | 下载原始输入 xlsx |
| `?download=simulate` | 下载仿真结果 CSV |
| `?download=evaluate` | 下载评分结果 CSV |
| `?download=report` | 下载报告（zip） |

## 输入文件格式

### 评测用例 xlsx

放置于 `data/inputs/`，Sheet 名 `ParsedData`，至少包含以下列：

| 列名 | 说明 |
|------|------|
| `task_id` | 用例唯一标识 |
| `query` | 用户首轮查询 |
| `context` | 上下文信息（JSON 字符串，含 uid/time/location 等） |
| `full_intent` | 用户完整意图描述 |
| `ground_truth` | 标注答案（JSON 字符串，含 explicit_intent/implicit_intent/truth_trajectory 等） |

## 输出文件结构

### 仿真结果 CSV

路径：`data/outputs/simulator_res/dialogue_<name>_<timestamp>_<model>.csv`

每行一个对话轮次：

| 列名 | 说明 |
|------|------|
| `conversation_id` | 对话 ID（= task_id） |
| `turn_index` | 轮次序号 |
| `role` | `user` / `assistant` |
| `content` | 对话内容 |
| `tool_calls` | 工具调用列表（JSON） |
| `is_stop` | 是否自然终止 |
| `execution_time_ms` | 本轮耗时 |
| `input_tokens` / `output_tokens` | Token 用量 |
| `ground_truth` | 原始标注 |

### 评分结果 CSV

路径：`data/outputs/evaluation_res/evaluation_result_<timestamp>_<suffix>.csv`

每行一个 case：

| 列名 | 说明 |
|------|------|
| `case_id` | 用例 ID（= conversation_id） |
| `status` | `success` / `error` / `skipped` |
| `error` | 错误信息（成功时为空） |
| `parse_errors` | 工具调用解析异常（JSON） |
| `results` | 评分结果（JSON，含 `metrics` + `details`） |
| `reason` | LLM 生成的评分理由（Markdown） |

`results.metrics` 结构：

```json
{
  "ECR": 1.0,
  "TS": 0.8,
  "IFS": 1.0,
  "IISR": 0.75,
  "AR": 0.75,
  "Eff": 0.55
}
```

### 评测报告

路径：`data/outputs/report/single/<timestamp>/`

| 文件 | 说明 |
|------|------|
| `summary.json` | 结构化聚合数据 |
| `summary.txt` | 文本报告 |
| `summary.html` | HTML 可视化报告 |
| `summary_charts.xlsx` | 指标图表（Excel） |
| `charts/*.png` | 指标分布图 |

多模型对比报告路径：`data/outputs/report/compare/<timestamp>/`

| 文件 | 说明 |
|------|------|
| `report.md` | Markdown 对比报告 |
| `comparison_summary.json` | 结构化对比数据 |

## 评测指标

### 质量指标

| 指标 | 全称 | 说明 | 范围 |
|------|------|------|------|
| **ECR** | Explicit-decision-factor Completion Rate | 显式意图完成率 | [0, 1] |
| **TS** | Tool Selection Accuracy | 工具选择正确性 | [0, 1] |
| **IFS** | Information Faithfulness Score | 信息忠实度（基于 factual_answer_rubric 行级判定） | [0, 1] |
| **IISR** | Implicit-decision-factor Satisfaction Rate | 隐式意图满足率（加权 Σ(Wi·Ci)/Σ(Wi)） | [0, 1] |
| **AR** | Accepted-response Probability | 综合达成率 = ECR × IISR | [0, 1] |
| **Eff** | Interaction Efficiency | 交互效率 = S_median / (S_median + S_actual)；0.5 = 人类基线，>0.5 比人类高效，<0.5 比人类冗余 | (0, 1) |
| **SES** | Satisfaction Efficiency Score | 综合效能 = AR × Eff；0.5 = 人类基线，>0.5 优于人类 | [0, 1] |
| **CEI** | Cost Efficiency Index | 性价比 = (SES / TotalTokens) × 10^6；越高越优 | [0, +∞) |

> **Eff 特殊处理**：当 ground_truth 中 `clarification_policy.max_allowed=0` 时，该 case 的 Eff 标记为 `skipped`，不参与批次 Eff/SES/CEI 均值计算。

### 运行时指标

| 指标 | 说明 | 聚合方式 |
|------|------|---------|
| **E2E_Latency** (ms) | 每轮 assistant 响应的端到端延迟均值 | mean / p50 / p95 / p99 |
| **InputTokens** | session 级 prompt tokens 总和 | 算术平均 |
| **OutputTokens** | session 级 completion tokens 总和 | 算术平均 |
| **TotalTokens** | input + output tokens | 算术平均 |

### 分母策略

- **质量指标**（ECR / TS / IFS / IISR / AR / SES / CEI）均值分母 = 全部 case（失败 case 按 0 分计入）
- **运行时指标**（E2E_Latency / Tokens）均值分母 = 仅 success case（避免物理量被 0 污染）

## 沙箱模式

沙箱模式下工具不调用真实 API，改为匹配预录制的 mock 数据：

1. **精确匹配**：`data/sandbox/mock_data/<tool>.csv` 按参数子集匹配
2. **向量模糊匹配**：文本参数（如 `query`）用 embedding 余弦相似度匹配（阈值 0.8）
3. **LLM 仿造**：均未命中时，由 `tool_simulator` 参考工具描述生成模拟响应

沙箱数据格式（CSV，UTF-8 BOM）：

| 列名 | 说明 |
|------|------|
| `trace_id` | 关联的 task_id |
| `tool_name` | 工具名 |
| `arguments` | 调用参数（JSON） |
| `result` | 返回结果（JSON） |
| `feature` | 模糊匹配用的 embedding 向量（可选） |

## 工具集

共 22 个工具，涵盖地图出行全场景：

| 工具名 | 说明 |
|------|------|
| `search_poi` | POI 关键词搜索 |
| `search_around_poi` | 周边 POI 搜索 |
| `search_poi_along_route` | 沿途 POI 搜索 |
| `search_poi_around_multipoints` | 多锚点周边搜索 |
| `get_navigation` | 驾车/步行/骑行路线规划 |
| `get_sequential_navigation` | 多点顺序导航 |
| `get_taxi_route_plan` | 打车路线与费用 |
| `get_rgeo` | 逆地理编码 |
| `get_weather` | 天气查询 |
| `get_route_traffic_info` | 路况查询 |
| `route_station_info` | 公交/地铁线路站点 |
| `optimize_visit_order` | 多点游览顺序优化 |
| `search_products_by_poiid` | POI 商品查询 |
| `search_train_or_flights_tickets` | 火车票/机票查询 |
| `fuel_payment` | 加油支付 |
| `restaurant_group_buy` | 餐厅团购 |
| `restaurant_reservation` | 餐厅预订 |
| `scenic_ticket_transaction` | 景区门票 |
| `transaction_service` | 交易服务 |
| `ainative_kuake_search` | 搜索引擎 |
| `search_user_action_summary` | 用户行为摘要 |
| `search_user_profile` | 用户画像 |

## 测试

```bash
uv run pytest                                        # 全部测试
uv run pytest tests/                                 # 集成测试
uv run pytest app/core/evaluation/tests/             # 指标层单测
```
