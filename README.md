### db-evaluator-agent

This repo contains **AgentBeats-compatible A2A agents** and utilities for evaluating database-related agent capabilities, plus a **Spider2-Snow SQL evaluation suite**.

### Repo layout

- **`evaluator/src/evaluation_suite/`**: Spider2-Snow evaluation scripts + dataset JSONL + gold execution results
  - **`evaluate.py`**: runs evaluation in `sql` mode (executes predicted SQL on Snowflake) or `exec_result` mode (compares predicted CSVs)
- **`evaluator/src/spider2sql_evaluator/`**: **A2A green agent** that benchmarks a target (purple) agent on Spider2-Snow SQL generation
  - `server.py`: exposes the A2A server + `AgentCard`
  - `executor.py`: task lifecycle + per-context agent state
  - `agent.py`: loads Spider2 tasks, calls target agent for SQL, runs evaluation, emits a result artifact
  - `messenger.py`: A2A client helper to talk to the target agent

### Spider2SQL benchmark (A2A evaluator agent)

The Spider2SQL evaluator is a **green agent** that evaluates a target agent by:

- Loading tasks from `evaluator/src/evaluation_suite/spider2-snow.dev25.jsonl` (dev) or `spider2-snow.jsonl` (full)
- Sending each task (`instance_id`, `instruction`, `db_id`, optional `external_knowledge`) to the target agent
- Expecting the target agent to return an A2A artifact **`DataPart` JSON** like:
  - `{"sql": "SELECT ..."}`
  - (text fallback is supported, including fenced ```sql blocks)
- Executing the SQL on **Snowflake** and comparing results to gold execution outputs using the evaluation suite
- Returning an A2A artifact named `Result` containing:
  - a human-readable summary (TextPart)
  - machine-readable metrics + per-instance results (DataPart)
