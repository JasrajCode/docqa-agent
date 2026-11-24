from typing import Literal

from pydantic import AwareDatetime, BaseModel

DocumentStatus = Literal["PROCESSING", "READY", "FAILED"]


class ChunkHit(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    chunk_index: int
    similarity: float
    content: str


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    chunk_index: int
    content: str


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    status: DocumentStatus
    chunk_count: int
    created_at: AwareDatetime
