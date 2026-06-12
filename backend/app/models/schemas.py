"""Pydantic API sxemalari."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


Scope = Literal["laws", "reports", "uploads"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    scope: Optional[list[Scope]] = None  # default: ["laws", "reports"]
    doc_id: Optional[str] = None
    use_cache: bool = True


class SourceItem(BaseModel):
    n: int
    source: str
    section: Optional[str] = None
    doc_id: Optional[str] = None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    from_cache: bool = False


class IngestResponse(BaseModel):
    doc_id: Optional[str]
    filename: Optional[str] = None
    chunks: int = 0
    collection: Optional[str] = None
    skipped: bool = False


class CompareRequest(BaseModel):
    doc_id_a: str
    doc_id_b: str
    aspects: Optional[list[str]] = None  # masalan: ["summasi", "muddati", "tomonlar"]


class CompareResponse(BaseModel):
    summary: str
    differences: list[str]
    sources: list[SourceItem]
