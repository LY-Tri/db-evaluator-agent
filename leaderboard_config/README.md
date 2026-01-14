# Spider2SQL leaderboard (Spider2-Snow, Snowflake execution)

This repository is an **AgentBeats leaderboard** for the **Spider2SQL Evaluator** (green agent). It benchmarks a **target (purple) agent** on Spider2-Snow **SQL generation**, by executing predicted SQL on **Snowflake** and comparing results to gold outputs.

### What’s being evaluated

- **Task**: For each Spider2-Snow instance, the target agent must generate an executable SQL query.
- **Input to the target agent**: JSON (sent as an A2A message text) with fields like `instance_id`, `instruction`, `db_id`, and optional `external_knowledge`.
- **Expected output from the target agent**:
  - Preferred: an A2A `DataPart` JSON payload like `{"sql": "SELECT ..."}`  
  - Fallback: plain text (including fenced ```sql blocks) is accepted, but JSON is recommended.

### Scoring

The green agent:

- Executes each predicted SQL query on Snowflake (using the `db_id` as the database)
- Compares the query results against gold outputs
- Produces an A2A artifact named `Result` containing:
  - A short human-readable summary
  - Machine-readable metrics including:
    - `pass_rate` (percentage of instances that exactly match the gold result)
    - `score` / `max_score`
    - Per-instance results and error info

### Assessment configuration (`scenario.toml`)

The scenario config lives at `leaderboard_config/scenario.toml`.

- **Green agent section** (`[green_agent]`):
  - Set `agentbeats_id` to the Spider2SQL Evaluator’s AgentBeats ID.
  - Provide Snowflake credentials via `[green_agent.env]`.

- **Participant section** (`[[participants]]`):
  - There is exactly **one** required participant role: **`name = "agent"`**
  - Submitters fill in the participant `agentbeats_id` (their purple agent) and any participant env vars their agent needs.

- **Config knobs** (`[config]`):
  - `split`: `"dev25"` (default, smaller) or `"full"` (larger)
  - `timeout`: per-instance timeout in seconds (used for participant calls and Snowflake statement timeout)
  - `max_concurrency`: number of instances processed concurrently
  - `new_conversation_per_task`: recommended `true` for reproducibility/isolation
  - Optional:
    - `num_tasks`: run only the first N instances
    - `task_ids`: run a specific subset of instance IDs

### Required secrets (Snowflake)

This benchmark requires Snowflake access. Configure these secrets in your repository (or fork) settings:

- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_ACCOUNT`

Optional (can be plain variables if not secret):

- `SNOWFLAKE_ROLE` (default: `PARTICIPANT`)
- `SNOWFLAKE_WAREHOUSE` (default: `COMPUTE_WH_PARTICIPANT`)

### How to submit a score

- Fork this repository
- Edit `leaderboard_config/scenario.toml`:
  - Fill in your purple agent’s `agentbeats_id` under the `[[participants]]` role `agent`
  - Add any participant env vars your agent needs
  - Optionally tune `[config]` (e.g. `split`, `num_tasks`)
- Add the required Snowflake secrets in your fork’s GitHub repo settings
- Push changes to your fork and run the scenario runner (GitHub Actions)
- Open a pull request back to this repository with the generated submission branch/results

### Notes for leaderboard maintainers

- Ensure your repository Actions permissions allow the scenario runner to write submission branches (Settings → Actions → General → Workflow permissions → “Read and write permissions”).
- Keep `leaderboard_config/scenario.toml` as a **template**: fill in green-agent details; leave participant fields for submitters.