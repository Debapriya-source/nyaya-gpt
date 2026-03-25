"""FastAPI application for Nyaya-GPT legal AI assistant."""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat, providers
from app.services.tree_index import load_or_build_tree

load_dotenv()

logger = logging.getLogger("nyaya-gpt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build/load document tree indexes on startup."""
    logger.info("Loading document tree indexes...")
    app.state.constitution_tree = load_or_build_tree("constitution", "constitution.pdf")
    app.state.bns_tree = load_or_build_tree("bns", "BNS.pdf")
    logger.info("Backend ready.")
    yield


app = FastAPI(
    title="Nyaya-GPT API",
    description="Legal AI assistant API using ReAct + tree-based RAG",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(providers.router)


@app.get("/api/health")
async def health():
    """Return a simple health-check response."""
    return {"status": "ok"}
