# schemas.py
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(..., description="用户的提问")
    strategy: str = Field("naive", description="RAG 策略，可选: naive, multi_query, hyde, hybrid, ultimate")

class AskResponse(BaseModel):
    answer: str = Field(..., description="大模型结合文档生成的回答")