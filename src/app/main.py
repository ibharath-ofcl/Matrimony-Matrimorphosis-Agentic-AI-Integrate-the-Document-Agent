import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from pydantic import BaseModel

from app.telemetry import (  # first: providers must exist before LangChain runs
    RUN_STEPS, HTTP_DURATION, HTTP_REQUESTS, configure_logging, log, run_config, setup_telemetry,
    shutdown_telemetry,
)

setup_telemetry()
configure_logging()
HTTPXClientInstrumentor().instrument()  # outgoing HTTP (URL ingestion, Groq) becomes child spans

from app.agent import build_agent, charts_from, clean_answer, steps_from
from app.debug_routes import install_routes
from app.context import Ctx
from app.errors import DocError, http_status
from app.kb import files
from app.kb.store import KnowledgeBase
from app.persistence import open_checkpointer
from app.schemas import AgentRequest, AgentResponse, KBResponse, UploadResponse
from app.tools_s1 import CHARTS_DIR
from app.users import InvalidCredentials, UserStore, UsernameTaken


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.kb = KnowledgeBase()  # loads the embedder, opens Qdrant + SQLite
    async with open_checkpointer() as checkpointer:
        agent, tools = await build_agent(app.state.kb, checkpointer=checkpointer)
        app.state.agent, app.state.tools = agent, tools
        app.state.tool_names = [t.name for t in tools]
        log.info("startup_complete", tools=app.state.tool_names)
        yield
    shutdown_telemetry()


app = FastAPI(title="Agentic AI assessment - Document Agent (tools + knowledge base + MCP)", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics")
app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")
install_routes(app)  # PROVIDED: /health and the /debug/* endpoints the grader reads (src/app/debug_routes.py, locked)


@app.middleware("http")
async def observe(request: Request, call_next):
    """Per-request: request_id in every log line, API-hit metrics by route/method/status."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=rid)
    t0, status = time.perf_counter(), 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        route = getattr(request.scope.get("route"), "path", request.url.path)  # template, not raw path
        elapsed = time.perf_counter() - t0
        attrs = {"http.route": route, "http.method": request.method, "http.status_code": str(status)}
        if route != "/health":
            HTTP_REQUESTS.add(1, attrs)
            HTTP_DURATION.record(elapsed, attrs)
            log.info("http_request", route=route, method=request.method, status=status,
                     duration_ms=round(elapsed * 1000))


user_store = UserStore()


class UserAuthRequest(BaseModel):
    username: str
    password: str


async def get_current_user_and_token(request: Request) -> tuple[dict, str]:
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user = await asyncio.to_thread(user_store.user_for_token, token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user, token


@app.post("/users", status_code=201)
async def register(req: UserAuthRequest):
    try:
        user = await asyncio.to_thread(user_store.create, req.username, req.password)
        return user
    except InvalidCredentials as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UsernameTaken as exc:
        raise HTTPException(status_code=409, detail=f"Username '{req.username}' is already taken.") from exc


@app.post("/login")
async def login(req: UserAuthRequest):
    res = await asyncio.to_thread(user_store.login, req.username, req.password)
    if not res:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return res


@app.get("/me")
async def me(auth: tuple[dict, str] = Depends(get_current_user_and_token)):
    user, _ = auth
    return {"user_id": user["user_id"], "username": user["username"]}


@app.post("/logout", status_code=204)
async def logout(auth: tuple[dict, str] = Depends(get_current_user_and_token)):
    _, token = auth
    await asyncio.to_thread(user_store.revoke, token)
    return Response(status_code=204)


@app.delete("/users/me", status_code=204)
async def delete_me(request: Request, auth: tuple[dict, str] = Depends(get_current_user_and_token)):
    user, _ = auth
    await asyncio.to_thread(user_store.delete_user, user["user_id"])
    request.app.state.kb.reset(user["user_id"])
    return Response(status_code=204)


@app.post("/agent", response_model=AgentResponse)
async def agent(req: AgentRequest, request: Request, auth: tuple[dict, str] = Depends(get_current_user_and_token)):
    """The agent decides: Session 1 tools, ingest_document, search_knowledge_base (cites document and section), MCP notes."""
    user, token = auth
    allowed = await asyncio.to_thread(user_store.claim_thread, req.thread_id, user["user_id"])
    if not allowed:
        raise HTTPException(status_code=403, detail="Thread owned by another user")

    structlog.contextvars.bind_contextvars(thread_id=req.thread_id)
    result = await _guard(
        request.app.state.agent.ainvoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config=run_config(thread_id=req.thread_id),
            context=Ctx(session_id=req.thread_id, user_id=user["user_id"], token=token),
        )
    )
    msgs = result["messages"]
    # only this turn: everything after the last human message
    turn = msgs[max(i for i, m in enumerate(msgs) if m.type == "human"):]
    steps = steps_from(turn)
    RUN_STEPS.record(len(steps))
    log.info("agent_done", steps=len(steps), tools=[s["tool"] for s in steps], answer_chars=len(str(msgs[-1].content)))
    return AgentResponse(answer=clean_answer(str(msgs[-1].content)), steps=steps, charts=charts_from(turn))


@app.post("/upload", response_model=UploadResponse)
async def upload(request: Request, file: UploadFile = File(...), session_id: str = Form(...), auth: tuple[dict, str] = Depends(get_current_user_and_token)):
    """Store a file for this session. It is NOT indexed yet: the agent calls ingest_document(file_id)."""
    user, _ = auth
    allowed = await asyncio.to_thread(user_store.claim_thread, session_id, user["user_id"])
    if not allowed:
        raise HTTPException(status_code=403, detail="Thread owned by another user")

    data = await file.read(files.MAX_BYTES + 1)
    try:
        file_id, path = files.save_upload(session_id, file.filename or "file", data)
    except DocError as exc:
        raise HTTPException(status_code=422, detail=f"{exc.message} {exc.next_step}") from exc
    log.info("file_uploaded", filename=path.name.split("__", 1)[1], size=len(data))
    return UploadResponse(file_id=file_id, filename=path.name.split("__", 1)[1], size=len(data))


@app.get("/kb/{session_id}", response_model=KBResponse)
async def kb_state(session_id: str, request: Request, auth: tuple[dict, str] = Depends(get_current_user_and_token)):
    """What this user's knowledge base holds, and exactly what the agent is told about it."""
    user, _ = auth
    kb: KnowledgeBase = request.app.state.kb
    kb_id = user["user_id"]
    prof = kb.get_profile(kb_id) or {"overview": None, "keywords": []}
    return KBResponse(**prof, documents=kb.documents(kb_id), prompt_block=kb.prompt_block(kb_id))


@app.delete("/kb/{session_id}")
async def kb_reset(session_id: str, request: Request, auth: tuple[dict, str] = Depends(get_current_user_and_token)):
    user, _ = auth
    request.app.state.kb.reset(user["user_id"])
    return {"status": "cleared"}


async def _guard(awaitable):
    try:
        return await awaitable
    except Exception as exc:
        log.error("request_failed", error_type=type(exc).__name__, error=str(exc)[:1000])
        status, message, headers = http_status(exc)  # safe text: no internals leak to the client
        raise HTTPException(status_code=status, detail=message, headers=headers) from exc
