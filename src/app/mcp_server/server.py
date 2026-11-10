"""Task 9: the notes and tasks MCP server. Run by docker compose / CI as:  python src/app/mcp_server/server.py --http  (port 9000, path /mcp)

The storage is PROVIDED (taskstore.py: TaskStore), the user lookup is PROVIDED (app/users.py: UserStore.user_for_token).
You write the integration: register the eight tools below on `mcp`, and make every call run AS THE CALLING USER.
"""
import json
import os
import sys

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from app.mcp_server.taskstore import TaskError, TaskStore
from app.users import UserStore

mcp = FastMCP("tasks", host="0.0.0.0", port=int(os.getenv("MCP_PORT", "9000")))
store = TaskStore()


def current_user_id(ctx: Context) -> str:
    """Reads the Bearer token from the incoming request headers and returns the user_id."""
    request = getattr(ctx.request_context, "request", None) if ctx else None
    if not request:
        raise ToolError("ERROR[UNAUTHENTICATED]: missing request context. Next step: log in and retry.")
    headers = getattr(request, "headers", {})
    auth = headers.get("authorization") or headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise ToolError("ERROR[UNAUTHENTICATED]: missing or invalid authorization header. Next step: log in and retry.")
    token = auth.split(" ", 1)[1].strip()
    user = UserStore().user_for_token(token)
    if not user:
        raise ToolError("ERROR[UNAUTHENTICATED]: unknown or revoked token. Next step: log in and retry.")
    return user["user_id"]


@mcp.tool()
def add_note(title: str, body: str, ctx: Context) -> str:
    """Add a new private note for the authenticated user.

    Args:
        title: Short title for the note (up to 120 chars)
        body: Body text of the note (up to 2000 chars)
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.add_note(user_id, title, body))
    except TaskError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def list_notes(ctx: Context) -> str:
    """List all private notes for the authenticated user."""
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.list_notes(user_id))
    except TaskError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def delete_note(note_id: int, ctx: Context) -> str:
    """Delete a note by its ID for the authenticated user.

    Args:
        note_id: Numeric identifier of the note to delete
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.delete_note(user_id, note_id))
    except TaskError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def add_task(title: str, details: str = "", ctx: Context = None) -> str:
    """Add a new task for the authenticated user with status 'todo'.

    Args:
        title: Short title for the task (up to 120 chars)
        details: Optional additional details or context for the task
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.add_task(user_id, title, details))
    except TaskError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def list_tasks(status: str = "", ctx: Context = None) -> str:
    """List tasks for the authenticated user, optionally filtered by status.

    Args:
        status: Filter by status ('todo', 'in_progress', or 'done'); leave empty for all
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.list_tasks(user_id, status=status))
    except TaskError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def set_task_status(task_id: int, status: str, ctx: Context = None) -> str:
    """Update the status of an existing task for the authenticated user.

    Valid transitions: todo -> in_progress, in_progress -> done,
    in_progress -> todo, done -> in_progress.

    Args:
        task_id: Numeric identifier of the task
        status: Target status ('todo', 'in_progress', or 'done')
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.set_task_status(user_id, task_id, status))
    except TaskError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def delete_task(task_id: int, ctx: Context = None) -> str:
    """Delete a task by its ID for the authenticated user.

    Args:
        task_id: Numeric identifier of the task to delete
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.delete_task(user_id, task_id))
    except TaskError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def task_summary(ctx: Context = None) -> str:
    """Get a count summary of tasks by status ('todo', 'in_progress', 'done') for the user."""
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.summary(user_id))
    except TaskError as exc:
        raise ToolError(str(exc)) from exc


if __name__ == "__main__":
    # stdio cannot carry the Authorization header: the agent talks to this server over HTTP (MCP_URL)
    mcp.run(transport="streamable-http" if "--http" in sys.argv else "stdio")
