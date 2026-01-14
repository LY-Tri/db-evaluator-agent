FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser agent
USER agent
WORKDIR /home/agent/app

# Install dependencies (locked)
COPY pyproject.toml uv.lock README.md ./

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

# Copy only evaluator code + bundled evaluation suite assets.
# (This image does NOT include the db_agent.)
COPY evaluator evaluator

ENTRYPOINT ["uv", "run", "python", "evaluator/src/spider2sql_evaluator/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9010", "--card-url", "http://localhost:9010/"]
EXPOSE 9010