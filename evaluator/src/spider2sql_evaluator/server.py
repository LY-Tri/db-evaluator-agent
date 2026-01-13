import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill


def main() -> None:
    from executor import Executor

    parser = argparse.ArgumentParser(description="Run the Spider2SQL evaluator (green agent).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9010, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="spider2sql_evaluation",
        name="Spider2SQL Benchmark Evaluation",
        description="Evaluates target agents on Spider2-Snow SQL generation by executing SQL and comparing results.",
        tags=["benchmark", "evaluation", "spider2", "sql"],
        examples=[
            '{"participants": {"agent": "http://localhost:9019"}, "config": {"split": "dev25", "num_tasks": 5, "timeout": 60}}'
        ],
    )

    agent_card = AgentCard(
        name="Spider2SQL Evaluator",
        description="Spider2-Snow benchmark evaluator (SQL execution + result comparison).",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    uvicorn.run(server.build(), host=args.host, port=args.port, timeout_keep_alive=300)


if __name__ == "__main__":
    main()

