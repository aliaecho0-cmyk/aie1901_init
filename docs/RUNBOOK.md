# tau-bench Retail Harness — 运行与交接说明

本文档给下一个接手这个任务的人。读完应该能独立跑起来、看懂产物、并知道改哪里。

---

## 0. 最短路径

环境就位后，一条命令：

```powershell
.\.venv\Scripts\python.exe scripts/run_tau_bench_500.py
```

但这**只在下面三件事都成立时**才成立，否则脚本会在第一步就报错：

1. `.venv` 是 **Python 3.12**（不是系统自带的 3.14）
2. `../tau2-bench` 这个目录还在原位置（editable 安装依赖它的绝对路径）
3. 项目根目录有 `.env`，里面有可用的 `OPENAI_API_KEY`

---

## 1. 这套东西在做什么

输入是 `data/synthetic_tasks/tau_bench_retail/task_0 … task_499`，每条任务是一个电商客服场景（tau-bench retail train split 的 500 条），每个目录里三个文件：

| 文件 | 内容 |
|---|---|
| `task.json` | `task_id`、客户原话 `prompt`、`available_tools`（本项目忽略它）、步数/预算 |
| `rubric.json` | 评分标准；标准答案动作（GT actions）以 JSON 数组形式串在 `criteria[*].description` 里 |
| `metadata.json` | `source: tau_bench`，用于让 `run_episode` 分派到真实 runner |

**没有 user simulator。** 客户的话作为一条 `UserMessage` 一次性交给 agent，agent 自己查库、改库、提交，整个 episode 只跑一次对话。

**打分不是 LLM judge。** 跑完后用 tau2 官方 `EnvironmentEvaluator`：把 agent 的完整消息轨迹重放进一个全新的 retail 环境得到终态 DB，把 GT 动作重放进另一个环境得到期望 DB，两者 hash 相同即 `db_match=True`，reward 记为 1.0，否则 0.0。

**产物**：每条任务一个 jsonl 轨迹（含官方打分），外加一个汇总 `summary.json`。

---

## 2. 核心设计决策

| 决策点 | 选择 | 原因 |
|---|---|---|
| 动作集 | 只有 `observe` / `use_tool` / `submit` | 没有 user simulator，`draft_solution`、`check_rubric` 没有意义 |
| 工具注入 | 强制暴露官方 16 个工具 | 完全忽略 `task.json` 里的 `available_tools`（那里只写了答案用到的那一个写工具） |
| 循环 | 真动态循环，最多 15 步 | LLM 每次看到工具返回后自己决定继续调工具还是 `submit` |
| 打分类型 | `EvaluationType.ENV` | 只比 DB 终态；`ALL` 会无条件跑 `ActionEvaluator`，对这批数据没有额外信息 |
| communicate | 固定中性 `1.0` | 本地数据没有 communicate 标注，且 `ENV` 模式下它不参与乘积 |
| 终止 | `submit` → `TerminationReason.AGENT_STOP` | evaluator 只认这个终止原因 |
| 运行模式 | 独立 runner | 不经过 tau2 的 orchestrator / user simulator |

---

## 3. 数据流

```
data/synthetic_tasks/tau_bench_retail/task_N/
  task.json / rubric.json / metadata.json
        │
        │  load_real_task()          覆盖 available_tools → 16 个官方工具
        ▼
  harness Task  ──task_brief()──►  UserMessage（客户原话，包装成 agent 视角）
        │
        ▼
  run_real_episode()  ◄──────────►  LLMClient（任何 OpenAI 兼容端点）
        │   每步返回一个 JSON 动作
        │
        ├─ observe   → act_observe()                 只打印 DB 统计，不进 tau2 history
        ├─ use_tool  → act_use_tool()
        │                 └─ tau2 Environment.get_response(ToolCall)
        │                       └─ ToolMessage
        │              AssistantMessage(tool_calls=[ToolCall]) + ToolMessage 成对入 history
        └─ submit    → AssistantMessage(final)        结束
        │
        ▼
  build_simulation_run()  →  score_run()
        │                        └─ tau2 evaluate_simulation(EvaluationType.ENV)
        │                              ├─ 重放 agent 轨迹 → predicted DB hash
        │                              └─ 重放 GT 动作   → gold DB hash
        ▼
  logger.records  →  <out>/trajectories/<task_id>.jsonl
                  →  <out>/summary.json
```

---

## 4. 文件地图

### 本任务真正相关的

| 文件 | 作用 | 什么时候需要改 |
|---|---|---|
| `scripts/run_tau_bench_500.py` | 批量入口：遍历任务、每条一个干净环境、落盘、汇总 | 改并发、改输出字段、改筛选逻辑 |
| `scripts/run_tau_bench_500_temp02.py` | 温度对照组批量入口：固定模型温度 `0.2` | 跑温度变体 |
| `scripts/run_tau_bench_500_prompt_checklist.py` | System prompt 对照组批量入口：温度 `0.0`，加入执行检查清单 | 跑 prompt 变体 |
| `code/harness/agent.py` | `run_real_episode`：system prompt、动态循环、动作解析、打分回写 | 改提示词、改步数上限、改动作协议 |
| `code/harness/actions.py` | `Task` 数据类 + 所有 action 实现；`act_use_tool` 是唯一接真实环境的地方 | 改工具调用方式 |
| `code/harness/tau2_adapter.py` | 任务加载、16 工具清单、环境构造、GT 解析、tau2 `Task` 构造 | 换数据源、改工具集合 |
| `code/harness/tau2_scorer.py` | 官方 evaluator 封装 + predicted/gold hash 诊断 | 换打分口径（例如加 action 匹配） |
| `code/harness/trajectory_logger.py` | 每步记录、`finalize` 写 `tau2` 字段、jsonl 落盘 | 加轨迹字段 |
| `code/harness/util/llm_client.py` | OpenAI 兼容客户端，带重试和成本估算 | 换模型/端点 |
| `test_m1.py` / `test_m2.py` / `test_m3.py` | 三级自测 | 排查问题时用 |
| `convert_data.py` | 把 `tasks_train.jsonl` 转成 `data/synthetic_tasks/...` | 数据集重建时用 |

### 不相关（原有 RL 研究代码，跑本任务用不到）

`code/rl/*`、`code/reward/*`、`code/modules/*`、`examples/*`、`figures/*`、`docs/ARTIFACT.md`。

注意：`agent.py` 里仍然保留了 `run_episode` 的旧分支（非 tau_bench 任务走固定序列）。本任务只走 `run_real_episode`。

---

## 5. 环境准备

### 5.1 Python 版本是硬约束

tau2 要求 `>=3.12,<3.14`。这台机器系统 Python 是 3.14.6，**不能直接用**，必须单独建 3.12 的 venv。

```powershell
cd <project-root>
uv venv .venv --python 3.12
```

### 5.2 tau2-bench 必须一起交接

tau2 是以 editable 方式安装的：

```powershell
uv pip install --python .venv\Scripts\python.exe -e ..\tau2-bench
```

这会在 `site-packages` 里写一个 `.pth` 指向 `../tau2-bench` 的绝对路径。**把项目拷走时必须把同级的 `tau2-bench` 目录一起拷走**，或者在新机器上重新执行上面这条命令。否则 `import tau2` 直接失败，报 `ModuleNotFoundError: No module named 'tau2'`。

tau2-bench 目录里必须带 `data/`（retail 的 db.json 等）。

### 5.3 其余依赖

```powershell
uv pip install --python .venv\Scripts\python.exe "openai>=1.0.0,<3.0.0" "python-dotenv"
```

注意：装 tau2 时它可能把 `openai` 拉到 3.x，而本项目要求 `<3.0.0`。如果之后 `agent.py` 报 OpenAI SDK 相关的参数错误，先检查 `openai` 版本（当前可用组合是 `openai 2.54.0` + `tau2 2.0.0`）。

### 5.4 模型配置

项目根目录建 `.env`：

```ini
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.deepseek.com"
OPENAI_MODEL="deepseek-chat"
```

任何 OpenAI 兼容端点都可以，换成别的模型只改这三行。项目的 LLMClient 和运行入口会自动加载根目录 `.env`，并以 `.env` 中的值覆盖同名进程环境变量。

### 5.5 验证安装

```powershell
.venv\Scripts\python.exe -c "import tau2, openai; print('ok')"
```

---

## 6. 三级自测（建议按顺序跑）

| # | 命令 | 期望输出 | 验证了什么 |
|---|---|---|---|
| 1 | `.venv\Scripts\python.exe test_m1.py` | `tool_count: 16`，`db_stats: {... num_orders: 1000 ...}`，`db_hash: b25c9cb2…` | 环境、16 工具、真实 DB 加载 |
| 2 | `$env:PYTHONPATH="$PWD\code"; .venv\Scripts\python.exe -m harness.tau2_adapter` | `task dirs: 500`，`gold actions: 500 tasks OK, total=768` | 500 条任务和 GT 动作全部可解析 |
| 3 | `.venv\Scripts\python.exe test_m3.py --mode all` | oracle `1.0`、skip-write `0.0`、tamper `0.0`、real `1.0` | 打分器有正负区分能力 |
| 4 | `.venv\Scripts\python.exe test_m2.py` | 打印逐步 `use_tool` 轨迹，末尾 `every ToolCall followed by matching ToolMessage: True` | 单条真实运行 + 消息契约 |

`test_m1.py` 输出的初始 `db_hash` 是全局初始库的指纹，`b25c9cb2…` 就代表"什么都没改"——如果某条任务跑完的 predicted hash 等于它，说明 agent 压根没写库。

`test_m3.py` 的四个模式可以单独跑：`--mode oracle|negative|tamper|real`。

---

## 7. 正式运行

### 7.1 先跑 5 条

```powershell
.venv\Scripts\python.exe scripts/run_tau_bench_500.py --limit 5 --out-dir runs/tau_bench_500_smoke
```

### 7.2 全量

```powershell
.venv\Scripts\python.exe scripts/run_tau_bench_500.py
```

默认输出到 `runs/tau_bench_500/`。

### 7.3 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--limit N` | 全部 | 只跑前 N 条 |
| `--start N` | `0` | 跳过前 N 条；`--start 250 --limit 250` 可分段跑 |
| `--out-dir PATH` | `runs/tau_bench_500` | 输出目录 |
| `--data-root PATH` | `data/synthetic_tasks/tau_bench_retail` | 任务根目录 |
| `--max-steps N` | `15` | 单条任务的最大步数 |
| `--task-retries N` | `1` | episode 以 `llm_error` 结束时的重试次数 |
| `--show-tool-logs` | 关 | 打开 tau2 每个工具的 DEBUG 日志（非常吵，排查时用） |
| `--quiet` | 关 | 不打印异常 traceback |

### 7.4 三组实验条件

每个入口使用同一批 retail 任务，但结果写入不同目录。已有 `runs/tau_bench_500/` 是 baseline；两个变体入口不会修改它。

| 条件 | 入口 | 温度 | System prompt | 默认输出目录 |
|---|---|---:|---|---|
| Baseline | `scripts/run_tau_bench_500.py` | `0.0` | 现有 prompt | `runs/tau_bench_500/` |
| 温度变体 | `scripts/run_tau_bench_500_temp02.py` | `0.2` | 现有 prompt | `runs/tau_bench_500_temp02/` |
| Prompt 变体 | `scripts/run_tau_bench_500_prompt_checklist.py` | `0.0` | 增加写前核实、必要写入、检查写入结果、确认后提交的执行清单 | `runs/tau_bench_500_prompt_checklist/` |

先各跑一条做 smoke，建议指定独立 smoke 输出目录：

```powershell
.venv\Scripts\python.exe scripts/run_tau_bench_500_temp02.py --limit 1 --out-dir runs/tau_bench_500_temp02_smoke
.venv\Scripts\python.exe scripts/run_tau_bench_500_prompt_checklist.py --limit 1 --out-dir runs/tau_bench_500_prompt_checklist_smoke
```

确认 smoke 的 summary 中 `counts.scored` 为 `1` 且有官方 reward 后，再启动全量。全量运行命令：

```powershell
.venv\Scripts\python.exe scripts/run_tau_bench_500_temp02.py
.venv\Scripts\python.exe scripts/run_tau_bench_500_prompt_checklist.py
```

两个入口都支持 `--limit`、`--start`、`--out-dir`、`--max-steps` 和 `--task-retries`。summary 的 `config` 会记录 `temperature` 与 `prompt_variant`。重跑前若要保留旧结果，请换一个 `--out-dir`；相同任务文件会在同一路径下被覆盖。

### 7.5 运行时行为

- 每条任务**新建一个** retail 环境，任务之间零状态残留；打分器内部也会各自新建环境。
- 单条任务抛异常不会中断整轮：该条记为 `error` 后继续。
- LLM 网络错误导致的 `llm_error` 会自动重试，默认 1 次。
- Ctrl+C 会停止循环，并写出 `interrupted: true` 的部分 `summary.json`。
- 终端每行打印一条进度：`[3/500] task_2 reward=1.00 db_match=True steps=4 tools=3 writes=1 submit 5.6s`

### 7.6 耗时与成本（参考）

5 条的实测：约 40 秒、估算 $0.05。按此推算 500 条约 **1 小时**、估算 **$4.8**。

`cost_usd` 是用 `llm_client.py` 里的 OpenAI 价格常数估算的，**不是** DeepSeek 的真实账单，只能当量级参考。真实费用以服务商账单为准。

---

## 8. 输出怎么读

### 8.1 目录结构

```
runs/tau_bench_500/
  trajectories/
    tau_retail_0.jsonl
    tau_retail_1.jsonl
    ...
  summary.json
```

### 8.2 `summary.json` 顶层字段

| 字段 | 含义 |
|---|---|
| `counts.requested / completed / errored / scored` | 请求数 / 正常完成 / 异常 / 成功拿到分数 |
| `mean_reward` | 官方 reward 的平均值（只在 `scored` 上算） |
| `db_match_rate` | DB 终态匹配率，等价于通过率 |
| `avg_tool_calls` | 平均 `use_tool` 次数 |
| `avg_write_calls` | 平均写库次数 |
| `avg_steps` | 平均总步数（含 observe / submit） |
| `avg_cost_usd` / `total_cost_usd` | 估算成本 |
| `reward_histogram` | 分数分布，例如 `{"0": 120, "1": 380}` |
| `termination_histogram` | 终止原因分布，重点看 `submit` 占比；`max_steps` 多说明模型在原地打转 |
| `elapsed_s` | 总墙钟时间 |
| `interrupted` | 是否被 Ctrl+C 中止 |
| `config` | 本次运行的参数快照 |
| `tasks[]` | 逐条明细 |

### 8.3 `tasks[]` 单条字段

`task_id`、`reward`、`db_match`、`termination_reason`、`steps`、`tool_calls`、`write_calls`、`invalid_actions`、`llm_errors`、`db_changed`、`predicted_db_hash`、`gold_db_hash`、`warnings`、`scoring_error`、`cost_usd`、`elapsed_s`、`trajectory_file`、`error`。

排查一条失败任务时，先看 `db_changed`：

- `db_changed=false` → agent 没写库（可能一直在问、或参数报错），看轨迹里的 `use_tool` 返回。
- `db_changed=true` 但 `db_match=false` → 写错了（订单/商品/数量/理由不对），对比 `predicted_db_hash` 和 `gold_db_hash`，再去轨迹里找那次写调用。

### 8.4 轨迹 jsonl

每条记录是一步，字段包括 `step`、`action`、`args`、`observation`（`status`、`summary`、`tool_name`、`cost`…）。

**最后一行的 `terminal` 记录**额外带：

| 字段 | 内容 |
|---|---|
| `terminal: true` | 标记 |
| `termination_reason` | `submit` / `max_steps` / `llm_error` |
| `tau2` | 官方打分完整结果：`reward`、`db_match`、`db_reward`、`communicate_reward`、`reward_basis`、`reward_breakdown`、`predicted_db_hash`、`gold_db_hash`、`warnings`、`official`（`RewardInfo` 原样 dump） |
| `final_rubric_score` | `score_source: "tau2_env"`，`rubric_score_norm` 就是官方 reward，旧口径留在 `legacy_rubric_score_norm` |
| `tau2_message_count` | 送入 evaluator 的消息条数 |

---

## 9. 改代码前必须知道的契约

1. **每个 `ToolCall` 必须放在 `AssistantMessage(tool_calls=[...])` 里，紧接着一条 `id` 相同的 `ToolMessage`。** 裸 `ToolMessage` 会让 `env.set_state` 直接抛错。
2. **客户 prompt 必须作为 `UserMessage` 进入 history**，不能塞进 system prompt——否则 evaluator 重放时对不上。
3. 只读工具在重放时被跳过，**写工具会重放并比对输出**，所以 `tool_args` 必须原样保留，不能事后改写。
4. 每条任务必须用全新环境（`build_retail_env()`），不要复用上一个任务的 env。
5. 正常结束必须是 `submit`，映射到 `TerminationReason.AGENT_STOP`；其他终止原因 evaluator 直接给 0。
6. 如果严格重放失败，`score_run` 会自动降级为非严格重放重试一次，原因写进 `warnings`。看到 `warnings` 非空要当成信号，不要忽略。

---

## 10. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'tau2'` | `../tau2-bench` 被移动/删除，或 venv 重建后没装 | 回到 5.2 重新 editable 安装 |
| `PackageNotFoundError: tau2-bench` | 包名是 `tau2` 不是 `tau2-bench`；查询时用 `tau2` | — |
| 各种语法/typing 报错，版本相关 | 用了系统 Python 3.14 | 确认 `python --version` 是 3.12，用 `.venv\Scripts\python.exe` 显式调用 |
| `OPENAI_API_KEY is not set` | 没有 `.env` 或未加载 | 检查根目录 `.env`；脚本会自动加载 |
| 401 / 403 | key 失效或 base_url 不对 | 换 key / 检查 `OPENAI_BASE_URL` |
| `reward: null` + `scoring_error` | evaluator 抛异常 | 看 `tau2.error`；常见原因是消息契约被破坏 |
| 大量 `termination_reason: llm_error` | 网络/限流 | 提高 `--task-retries`，或降低并发（本 runner 是串行的） |
| 大量 `max_steps` | 模型反复查询不提交 | 看轨迹，可能提示词需要加强"查到就动手"的约束 |
| 明明改对了库但 reward 0 | gold 动作解析或参数不一致 | 用 `test_m3.py --mode oracle` 验证基线是否为 1.0 |
| 终端被 tau2 日志刷屏 | 默认关闭 DEBUG，但 `--show-tool-logs` 会打开 | 去掉该参数 |

---

## 11. 已知限制

- **communicate 恒为 1.0**：本地数据没有 communicate 标注，这部分不提供信号。
- **没有 user simulator**：模型必须在一次阅读后理解完整需求，无法追问澄清。这对多约束任务偏难，也意味着分数不能直接和 tau-bench 官方榜对比。
- **只看 DB 终态**：不评价过程是否高效、是否多做了不该做的写操作（只要终态 hash 一致就算对）。
- **可能饱和**：5 条样本全 1.0，全量如果 `mean_reward` 接近 1.0，说明这批任务对该模型偏简单，需要看 `reward_histogram` 分布决定是否有区分度。
- **`data/` 不在 git 里**：`data/` 被 `.gitignore` 忽略，它是 `convert_data.py` 从 `tasks_train.jsonl` 生成的可再生产物；而 `tasks_train.jsonl` 也被 `*.jsonl` 规则忽略。交接时要么把这两样一起拷走，要么确认对方能重建。
- **`third_party/tau-bench` 未入库**：60MB 上游副本，项目代码没有引用它，已加入 `.gitignore`。
- **成本是估算值**，不是真实账单。

---

## 12. 交接前 checklist

- [ ] `../tau2-bench` 目录一起交接（或对方能自行获取并重装）
- [ ] `data/synthetic_tasks/tau_bench_retail/` 或 `tasks_train.jsonl` 一起交接
- [ ] 对方能在新机器上重建 venv 并跑通 6 节的三级自测
- [ ] **轮换 API key**：`.env.example` 和 `examples/anchor_2_base_harness.py` 里各有一个明文 key，且已经进入 git 历史。交接前应作废该 key、把这两处换成占位符；只改工作区文件不能消除历史记录。
- [ ] 确认 `.env` 没有被拷进交接包（它已被 `.gitignore` 忽略）
