from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    session_id: str
    message: str
    disease_program: Optional[str] = "FA"


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    status: str