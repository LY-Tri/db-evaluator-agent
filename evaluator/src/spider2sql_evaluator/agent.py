import asyncio
import importlib.util
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import nest_asyncio
from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

try:
    # Package import (preferred)
    from spider2sql_evaluator.messenger import Messenger
except Exception:  # pragma: no cover
    # Script execution fallback
    from messenger import Messenger

logger = logging.getLogger("spider2sql_evaluator")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# NOTE: Applying nest_asyncio inside a server process can be surprising and is
# usually only needed in notebook-like nested event-loop environments.
# Opt-in via env var if someone really needs it.
if str(os.environ.get("SPIDER2SQL_EVAL_NEST_ASYNCIO", "")).lower() in {"1", "true", "yes"}:
    nest_asyncio.apply()


# /.../repo/evaluator/src/spider2sql_evaluator/agent.py -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SUITE_DIR = REPO_ROOT / "evaluator" / "src" / "evaluation_suite"


class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class Spider2Task(BaseModel):
    instance_id: str
    instruction: str
    db_id: str
    external_knowledge: str | None = None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _select_tasks(
    *,
    split: str,
    task_ids: list[str] | None,
    num_tasks: int | None,
) -> list[Spider2Task]:
    if split not in {"dev25", "full"}:
        raise ValueError("config.split must be 'dev25' or 'full'")

    dataset_path = EVAL_SUITE_DIR / (
        "spider2-snow.dev25.jsonl" if split == "dev25" else "spider2-snow.jsonl"
    )
    all_rows = _load_jsonl(dataset_path)

    if task_ids:
        wanted = set(task_ids)
        all_rows = [r for r in all_rows if r.get("instance_id") in wanted]

    if num_tasks is not None:
        all_rows = all_rows[:num_tasks]

    return [Spider2Task.model_validate(r) for r in all_rows]


def _build_target_prompt(task: Spider2Task) -> str:
    # Target should respond with a DataPart {"sql": "..."}.
    payload = {
        "instance_id": task.instance_id,
        "instruction": task.instruction,
        "db_id": task.db_id,
        "external_knowledge": task.external_knowledge,
        "response_format": {"type": "json", "schema": {"sql": "string"}},
    }
    return json.dumps(payload)


def _extract_sql_from_target_response(outputs: dict[str, object]) -> str | None:
    # Preferred: DataPart JSON like {"sql": "..."} in outputs["data"].
    for item in outputs.get("data", []) if isinstance(outputs.get("data"), list) else []:
        if isinstance(item, dict) and isinstance(item.get("sql"), str) and item["sql"].strip():
            return str(item["sql"]).strip()

    # Fallback: parse text
    text = outputs.get("text")
    if isinstance(text, str) and text.strip():
        raw = text.strip()

        # If the target replied with JSON in text, accept {"sql": "..."}.
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and isinstance(obj.get("sql"), str) and obj["sql"].strip():
                return str(obj["sql"]).strip()
        except Exception:
            pass

        # If it's fenced SQL, strip to the contents.
        if "```" in raw:
            for fence in ("```sql", "```SQL", "```"):
                if fence in raw:
                    parts = raw.split("```")
                    if len(parts) >= 3:
                        candidate = parts[1]
                        candidate_lines = candidate.splitlines()
                        if candidate_lines and candidate_lines[0].strip().lower() == "sql":
                            candidate_lines = candidate_lines[1:]
                        return "\n".join(candidate_lines).strip() or raw

        return raw

    return None


def _load_evaluate_module():
    eval_path = (EVAL_SUITE_DIR / "evaluate.py").resolve()
    spec = importlib.util.spec_from_file_location("spider2_eval_suite_evaluate", str(eval_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {eval_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


class Agent:
    required_roles: list[str] = ["agent"]
    required_config_keys: list[str] = []

    def __init__(self):
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, validation_msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(validation_msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        cfg = request.config or {}
        split = str(cfg.get("split", "dev25"))
        task_ids = cfg.get("task_ids")
        # Normalize task_ids to list[str] if provided.
        if task_ids is not None:
            if not isinstance(task_ids, list):
                task_ids = [task_ids]
            task_ids = [str(x) for x in task_ids]
        num_tasks = cfg.get("num_tasks")
        num_tasks = int(num_tasks) if num_tasks is not None else None
        timeout = int(cfg.get("timeout", 60))
        max_concurrency = int(cfg.get("max_concurrency", 8))
        # By default, isolate each benchmark instance into a fresh conversation to
        # avoid cross-instance leakage and concurrency races.
        new_conversation_per_task = bool(cfg.get("new_conversation_per_task", True))
        if not new_conversation_per_task and max_concurrency > 1:
            # A shared conversation context cannot be safely used with concurrency.
            logger.warning(
                "new_conversation_per_task=false with max_concurrency>1; forcing max_concurrency=1 to avoid context races."
            )
            max_concurrency = 1

        agent_url = str(request.participants["agent"])

        start_time = time.time()
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Loading Spider2 tasks (split={split})..."),
        )

        tasks = _select_tasks(split=split, task_ids=task_ids, num_tasks=num_tasks)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of {len(tasks)} instances..."),
        )

        run_dir = Path(tempfile.mkdtemp(prefix="spider2sql_eval_"))
        pred_sql_dir = run_dir / "pred_sql"
        pred_sql_dir.mkdir(parents=True, exist_ok=True)

        instance_sql: dict[str, str] = {}
        instance_errors: dict[str, str] = {}

        sem = asyncio.Semaphore(max_concurrency)

        async def _run_one(t: Spider2Task) -> None:
            async with sem:
                try:
                    prompt = _build_target_prompt(t)
                    outputs = await self.messenger.talk_to_agent(
                        message=prompt,
                        url=agent_url,
                        new_conversation=new_conversation_per_task,
                        timeout=timeout,
                    )
                    sql_or_text = _extract_sql_from_target_response(outputs)
                    if not sql_or_text:
                        raise RuntimeError("Target agent returned no sql/text")

                    (pred_sql_dir / f"{t.instance_id}.sql").write_text(sql_or_text, encoding="utf-8")
                    instance_sql[t.instance_id] = sql_or_text
                except Exception as e:
                    instance_errors[t.instance_id] = str(e)
                    (pred_sql_dir / f"{t.instance_id}.sql").write_text("", encoding="utf-8")

        try:
            await asyncio.gather(*[_run_one(t) for t in tasks])
        finally:
            self.messenger.reset()

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Running evaluation suite (Snowflake execution + gold comparison)..."),
        )

        eval_mod = _load_evaluate_module()
        gold_dir = str((EVAL_SUITE_DIR / "gold").resolve())
        eval_result = eval_mod.run_spider2sql_evaluation(
            mode="sql",
            result_dir=str(pred_sql_dir),
            gold_dir=gold_dir,
            timeout=timeout,
            max_workers=max_concurrency,
            temp_dir=str(run_dir / "temp"),
            save_correct_ids=False,
            quiet=True,
        )

        time_used = time.time() - start_time
        score = float(eval_result["score"])
        max_score = int(eval_result["max_score"])
        pass_rate = float(eval_result["pass_rate"])

        per_instance: dict[str, Any] = dict(eval_result.get("instances", {}))
        for iid, err in instance_errors.items():
            if iid not in per_instance:
                per_instance[iid] = {
                    "instance_id": iid,
                    "score": 0,
                    "pred_sql": instance_sql.get(iid),
                    "error_info": err,
                }

        result_data = {
            "split": split,
            "num_instances": len(tasks),
            "score": score,
            "max_score": max_score,
            "pass_rate": pass_rate,
            "time_used_sec": time_used,
            "instances": per_instance,
            "run_dir": str(run_dir),
            "config": {
                "timeout": timeout,
                "max_concurrency": max_concurrency,
                "task_ids": task_ids,
                "num_tasks": num_tasks,
            },
        }

        failed = [iid for iid, r in per_instance.items() if float(r.get("score", 0)) != 1.0]
        failed_preview = "\n".join(f"- {iid}: {per_instance[iid].get('error_info')}" for iid in failed[:10])
        summary = (
            "Spider2SQL Benchmark Results\n"
            f"Split: {split}\n"
            f"Instances: {len(tasks)}\n"
            f"Pass Rate: {pass_rate:.1f}% ({int(score)}/{max_score})\n"
            f"Time: {time_used:.1f}s\n"
        )
        if failed_preview:
            summary += f"\nFirst failures:\n{failed_preview}\n"

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=summary)), Part(root=DataPart(data=result_data))],
            name="Result",
        )

