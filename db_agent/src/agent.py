import json

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message


class Agent:
    """
    Dummy target agent for Spider2SQL evaluation.

    Contract: return a DataPart JSON like {"sql": "..."}.
    """

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        _ = get_message_text(message)

        await updater.update_status(TaskState.working, new_agent_text_message("Returning dummy SQL..."))

        # Simple, always-valid query on Snowflake.
        # This will almost certainly not match gold results, but is good for pipeline smoke tests.
        payload = {"sql": "SELECT 1 AS DUMMY_VALUE;"}

        await updater.add_artifact(parts=[Part(root=DataPart(data=payload))], name="Action")

