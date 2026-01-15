import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


def _create_message(*, text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


async def _wait_for_agent(base_url: str, timeout_sec: float = 20.0) -> None:
    start = time.time()
    async with httpx.AsyncClient(timeout=5) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        while True:
            try:
                await resolver.get_agent_card()
                return
            except Exception:
                if time.time() - start > timeout_sec:
                    raise TimeoutError(f"Timed out waiting for agent at {base_url}")
                await asyncio.sleep(0.2)


async def _send_a2a_message(*, base_url: str, payload_text: str) -> str:
    async with httpx.AsyncClient(timeout=600) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=False)).create(agent_card)

        outbound = _create_message(text=payload_text)
        last_event = None
        async for event in client.send_message(outbound):
            last_event = event
        return str(last_event)


def _popen(cmd: list[str]) -> subprocess.Popen:
    env = os.environ.copy()
    return subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _terminate(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    try:
        p.send_signal(signal.SIGTERM)
    except Exception:
        return
    try:
        p.wait(timeout=5)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


async def main() -> int:
    # Load repo-root .env (if present) so Snowflake credentials can be provided via env vars.
    load_dotenv()

    evaluator_url = "http://127.0.0.1:9010"
    target_url = "http://127.0.0.1:9019"

    target = _popen([sys.executable, "db_agent/src/server.py", "--host", "127.0.0.1", "--port", "9019"])
    evaluator = _popen(
        [sys.executable, "evaluator/src/spider2sql_evaluator/server.py", "--host", "127.0.0.1", "--port", "9010"]
    )

    try:
        await _wait_for_agent(target_url)
        await _wait_for_agent(evaluator_url)

        req = {
            "participants": {"agent": target_url},
            "config": {"split": "dev25", "num_tasks": 1, "timeout": 5, "max_concurrency": 1},
        }
        payload_text = json.dumps(req)

        print("Sending evaluation request to evaluator...")
        resp = await _send_a2a_message(base_url=evaluator_url, payload_text=payload_text)
        print(resp)
        return 0
    finally:
        _terminate(evaluator)
        _terminate(target)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

