"""Chat endpoint with SSE streaming."""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest
from app.services.agent import run_agent_stream

router = APIRouter()


@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    """Stream a ReAct agent response for the given chat message via SSE."""
    return StreamingResponse(
        run_agent_stream(
            query=req.message,
            provider=req.provider,
            ollama_model=req.ollama_model,
            constitution_tree=request.app.state.constitution_tree,
            bns_tree=request.app.state.bns_tree,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
