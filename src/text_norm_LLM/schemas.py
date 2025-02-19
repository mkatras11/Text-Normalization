from pydantic import BaseModel, Field
from typing import List, Optional

# Pydantic Models for Structured Output
class NormalizedText(BaseModel):
    """Represents the normalized text output from the OpenAI API."""
    raw_comp_writers_text: Optional[str] = Field(..., description="Raw composer/writer names")
    CLEAN_TEXT: Optional[str] = Field(..., description="Normalized composer/writer names")


class NormalizedTextResponse(BaseModel):
    """Represents the normalized text response for multiple inputs."""
    normalized_text: List[NormalizedText] = Field(..., description="List of normalized text outputs")