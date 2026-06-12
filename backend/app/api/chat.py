"""Chat / savol-javob endpointlari."""
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core import rag
from app.models.schemas import AskRequest, AskResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        result = await rag.ask(
            question=req.question,
            scope=req.scope,
            doc_id=req.doc_id,
            use_cache=req.use_cache,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """Server-Sent Events (SSE) javob qaytaradi."""

    async def event_gen():
        try:
            async for chunk in rag.ask_stream(
                question=req.question, scope=req.scope, doc_id=req.doc_id
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
