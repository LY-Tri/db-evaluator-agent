import json
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory, Consumer
from a2a.types import DataPart, Message, Part, Role, TextPart


DEFAULT_TIMEOUT = 300


def create_message(*, role: Role = Role.user, text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def _collect_parts(parts: list[Part]) -> tuple[str, list[object]]:
    text_chunks: list[str] = []
    data_parts: list[object] = []
    for part in parts:
        if isinstance(part.root, TextPart):
            text_chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            data_parts.append(part.root.data)
    return "\n".join(text_chunks), data_parts


async def send_message(
    *,
    message: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    consumer: Consumer | None = None,
) -> dict[str, object]:
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        client = ClientFactory(config).create(agent_card)
        if consumer:
            await client.add_event_consumer(consumer)

        outbound_msg = create_message(text=message, context_id=context_id)
        last_event = None

        async for event in client.send_message(outbound_msg):
            last_event = event

        outputs: dict[str, object] = {"context_id": None, "status": "completed", "text": "", "data": []}

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                txt, data = _collect_parts(msg.parts)
                outputs["text"] = txt
                outputs["data"] = data
            case (task, _update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value

                status_msg = task.status.message
                if status_msg:
                    txt, data = _collect_parts(status_msg.parts)
                    outputs["text"] = str(outputs["text"]) + txt
                    outputs["data"] = list(outputs["data"]) + data

                if task.artifacts:
                    for artifact in task.artifacts:
                        txt, data = _collect_parts(artifact.parts)
                        if txt:
                            outputs["text"] = str(outputs["text"]) + ("\n" if outputs["text"] else "") + txt
                        outputs["data"] = list(outputs["data"]) + data
            case _:
                pass

        return outputs


class Messenger:
    def __init__(self):
        self._context_ids: dict[str, str | None] = {}

    async def talk_to_agent(
        self,
        *,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> dict[str, object]:
        outputs = await send_message(
            message=message,
            base_url=url,
            context_id=None if new_conversation else self._context_ids.get(url),
            timeout=timeout,
        )
        if outputs.get("status", "completed") != "completed":
            raise RuntimeError(f"{url} responded with: {json.dumps(outputs, indent=2)}")
        self._context_ids[url] = str(outputs.get("context_id")) if outputs.get("context_id") else None
        return outputs

    def reset(self) -> None:
        self._context_ids = {}

