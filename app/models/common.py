from pydantic import BaseModel, Field
from typing import Any


class BaseResponse(BaseModel):
    message: str = Field(..., description="Response message")
    code: int = Field(200, description="Response status code")
    data: Any = Field(None, description="Response data payload")
