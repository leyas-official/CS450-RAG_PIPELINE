from typing import List, Optional
from typing import Literal
from pydantic import BaseModel
from dataclasses import dataclass, field


class Source(BaseModel):
    document: str
    section: Optional[str]
    text_excerpt: str


class Requirement(BaseModel):
    requirement_id: str
    purpose: str
    actor: str
    action: str
    conditions: str
    priority: Optional[Literal["high", "medium", "low"]] = field(default="medium")


class RequirementsOutput(BaseModel):
    requirements: List[Requirement]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    article_numbers: List[int] = field(default_factory=list)
    effective_mode: str = None
    file_name: str = None