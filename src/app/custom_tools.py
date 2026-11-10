"""Tasks 7 and 8: four tools. Names, arguments, outputs, errors, metric and log names are FIXED (the autograder checks them).

The heavy parts are PROVIDED (kb/store.py: stats / outline / delete_document, kb/pipeline.py: ingest_file).
You write the tool wrappers: signature + docstring (the docstring is the prompt: say WHEN to use it, describe each argument),
validation, ToolException with `ERROR[CODE]: what happened. Next step: what to do.`, the metric, the log line, `handle_tool_error = True`.
All four are `async def` (blocking calls go through asyncio.to_thread). They act on the CALLING USER's knowledge base:
`runtime.context.user_id` (task 6), never on another user's. build_custom_tools(kb) returns all four.
Look at kb/tools.py for the pattern (ToolRuntime, ToolException, handle_tool_error).
"""
import asyncio
import time

from langchain.tools import ToolRuntime
from langchain_core.tools import ToolException, tool
from opentelemetry import metrics

from app.context import Ctx
from app.errors import to_tool_exception
from app.kb import files
from app.kb.pipeline import ingest_file
from app.kb.store import KnowledgeBase
from app.telemetry import log

_meter = metrics.get_meter("agent")
KBSTATS_CALLS = _meter.create_counter("kbstats.calls", unit="1", description="Calls to kb_stats")
OUTLINE_CALLS = _meter.create_counter("outline.calls", unit="1", description="Calls to document_outline")


def build_custom_tools(kb: KnowledgeBase) -> list:
    @tool
    async def kb_stats(runtime: ToolRuntime[Ctx], top_keywords: int = 5) -> str:
        """Get summary statistics and top keywords for the user's knowledge base.
        Use when the user asks what documents are in the knowledge base, how many
        documents, sections, or chunks exist, or what main topics/keywords are covered.

        Args:
            top_keywords: number of top keywords to return (between 1 and 20, default 5)
        """
        t0 = time.perf_counter()
        user_id = getattr(runtime.context, "user_id", None) or runtime.context.session_id
        if not isinstance(top_keywords, int) or top_keywords < 1 or top_keywords > 20:
            duration_ms = round((time.perf_counter() - t0) * 1000)
            KBSTATS_CALLS.add(1, {"status": "error"})
            log.info("kb_stats_done", tool="kb_stats", status="error", duration_ms=duration_ms)
            raise ToolException("ERROR[INVALID_INPUT]: top_keywords must be an integer between 1 and 20. "
                                "Next step: retry with top_keywords between 1 and 20.")
        try:
            s = await asyncio.to_thread(kb.stats, user_id)
            if not s["documents"]:
                result = "The knowledge base is empty."
            else:
                kws = ", ".join(s["keywords"][:top_keywords])
                result = f"{s['documents']} documents, {s['sections']} sections, {s['chunks']} chunks. Top keywords: {kws}."
            duration_ms = round((time.perf_counter() - t0) * 1000)
            KBSTATS_CALLS.add(1, {"status": "ok"})
            log.info("kb_stats_done", tool="kb_stats", status="ok", duration_ms=duration_ms)
            return result
        except ToolException:
            raise
        except Exception as exc:
            duration_ms = round((time.perf_counter() - t0) * 1000)
            KBSTATS_CALLS.add(1, {"status": "error"})
            log.info("kb_stats_done", tool="kb_stats", status="error", duration_ms=duration_ms)
            raise to_tool_exception(exc) from exc

    @tool
    async def document_outline(doc_name: str, runtime: ToolRuntime[Ctx]) -> str:
        """Get the structural outline (sections, headings, page numbers, summaries) of one document.
        Use when the user asks what a document contains, how it is organised, or which section
        covers a topic.

        Args:
            doc_name: name or title of the document to outline
        """
        t0 = time.perf_counter()
        user_id = getattr(runtime.context, "user_id", None) or runtime.context.session_id
        try:
            exact = await asyncio.to_thread(kb.resolve_doc_name, user_id, doc_name)
            if exact is None:
                docs = await asyncio.to_thread(kb.documents, user_id)
                names = ", ".join(f'"{d["filename"]}"' for d in docs) or "(none)"
                raise ToolException(f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}'. "
                                    f"Documents in your knowledge base: {names}. "
                                    "Next step: retry with one of these names.")
            docs = await asyncio.to_thread(kb.documents, user_id)
            doc = next((d for d in docs if d["filename"] == exact), None)
            if not doc:
                raise ToolException(f"ERROR[UNKNOWN_DOCUMENT]: Document '{exact}' not found. Next step: retry with a valid document name.")
            sections = await asyncio.to_thread(kb.outline, user_id, doc["doc_id"])
            lines = [f"Outline of '{exact}' ({len(sections)} sections):"]
            for i, s in enumerate(sections, start=1):
                p_start, p_end = s.get("page_start"), s.get("page_end")
                if p_start is not None:
                    pages = f" (p.{p_start})" if p_start == p_end else f" (p.{p_start}-{p_end})"
                else:
                    pages = ""
                level = s.get("level", 1)
                heading = s.get("heading", "")
                summary = s.get("summary", "")
                lines.append(f"{i}. [L{level}] {heading}{pages}: {summary}")
            result = "\n".join(lines)
            duration_ms = round((time.perf_counter() - t0) * 1000)
            OUTLINE_CALLS.add(1, {"status": "ok"})
            log.info("document_outline_done", tool="document_outline", status="ok", duration_ms=duration_ms, sections=len(sections))
            return result
        except ToolException:
            duration_ms = round((time.perf_counter() - t0) * 1000)
            OUTLINE_CALLS.add(1, {"status": "error"})
            log.info("document_outline_done", tool="document_outline", status="error", duration_ms=duration_ms, sections=0)
            raise
        except Exception as exc:
            duration_ms = round((time.perf_counter() - t0) * 1000)
            OUTLINE_CALLS.add(1, {"status": "error"})
            log.info("document_outline_done", tool="document_outline", status="error", duration_ms=duration_ms, sections=0)
            raise to_tool_exception(exc) from exc

    @tool
    async def delete_document(doc_name: str, runtime: ToolRuntime[Ctx]) -> str:
        """Removes one document from the user's knowledge base. Use when the user asks to
        delete or remove a document.

        Args:
            doc_name: exact file name or close match from the KNOWLEDGE BASE list
        """
        user_id = getattr(runtime.context, "user_id", None) or runtime.context.session_id
        exact = await asyncio.to_thread(kb.resolve_doc_name, user_id, doc_name)
        if exact is None:
            docs = await asyncio.to_thread(kb.documents, user_id)
            names = ", ".join(f'"{d["filename"]}"' for d in docs) or "(none)"
            raise ToolException(f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}'. "
                                f"Documents in your knowledge base: {names}. "
                                "Next step: retry with one of these names.")
        docs = await asyncio.to_thread(kb.documents, user_id)
        doc = next((d for d in docs if d["filename"] == exact), None)
        if not doc:
            raise ToolException(f"ERROR[UNKNOWN_DOCUMENT]: Document '{exact}' not found. Next step: retry with a valid document name.")
        await asyncio.to_thread(kb.delete_document, user_id, doc["doc_id"])
        remaining = len(await asyncio.to_thread(kb.documents, user_id))
        return f"Deleted '{exact}'. {remaining} documents left."

    @tool
    async def update_document(doc_name: str, source: str, runtime: ToolRuntime[Ctx]) -> str:
        """Replaces an existing document with a new version the user uploaded.

        Args:
            doc_name: file name of the document to replace
            source: the file_id of the new upload
        """
        user_id = getattr(runtime.context, "user_id", None) or runtime.context.session_id
        session_id = runtime.context.session_id
        exact = await asyncio.to_thread(kb.resolve_doc_name, user_id, doc_name)
        if exact is None:
            docs = await asyncio.to_thread(kb.documents, user_id)
            names = ", ".join(f'"{d["filename"]}"' for d in docs) or "(none)"
            raise ToolException(f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}'. "
                                f"Documents in your knowledge base: {names}. "
                                "Next step: retry with one of these names.")
        docs = await asyncio.to_thread(kb.documents, user_id)
        old_doc = next((d for d in docs if d["filename"] == exact), None)
        if not old_doc:
            raise ToolException(f"ERROR[UNKNOWN_DOCUMENT]: Document '{exact}' not found. Next step: retry with a valid document name.")

        try:
            path, _ = files.resolve(session_id, source.strip())
        except Exception as exc:
            raise to_tool_exception(exc) from exc

        try:
            result, existing = await ingest_file(kb, user_id, path, exact, runtime.config)
        except Exception as exc:
            raise to_tool_exception(exc) from exc

        if existing:
            return f"'{exact}' is already up to date."

        if result.doc_id != old_doc["doc_id"]:
            await asyncio.to_thread(kb.delete_document, user_id, old_doc["doc_id"])

        return f"Updated '{exact}': {len(result.sections)} sections, {result.n_chunks} chunks."

    for t in (kb_stats, document_outline, delete_document, update_document):
        t.handle_tool_error = True
    return [kb_stats, document_outline, delete_document, update_document]
