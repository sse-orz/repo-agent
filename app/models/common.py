from pydantic import BaseModel, Field
from typing import Optional, Any


class BaseResponse(BaseModel):
    message: str = Field(..., description="Response message")
    code: Optional[int] = Field(200, description="Response status code")
    data: Optional[Any] = Field(None, description="Response data payload")
