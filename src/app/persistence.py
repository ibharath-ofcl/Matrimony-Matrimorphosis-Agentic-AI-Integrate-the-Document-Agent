"""Chat memory: the LangGraph checkpointer that remembers each conversation (thread_id).

TODO(task 3): chat memory must survive a restart of the API container.
  Today it is `InMemorySaver`: restart the container and every conversation is forgotten.
  Replace it with a SQLite-backed saver that stores to the file named by CHECKPOINT_DB.

Hints
  - the package `langgraph-checkpoint-sqlite` is already installed
  - the agent runs async, so you want the async saver; `AsyncSqliteSaver.from_conn_string(path)` is an async CONTEXT MANAGER
  - open it once at startup and keep it open until shutdown (the lifespan in main.py holds it open)
  - make sure the folder of CHECKPOINT_DB exists
  - then integrate it: main.py's lifespan (TODO task 3) opens it and agent.py's build_agent receives it
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")


@asynccontextmanager
async def open_checkpointer():
    path = Path(CHECKPOINT_DB).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver
