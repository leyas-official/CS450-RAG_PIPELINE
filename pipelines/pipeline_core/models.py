from typing import List, Optional
from pydantic import BaseModel
from dataclasses import dataclass, field


class Source(BaseModel):
    document: str
    section: Optional[str]
    text_excerpt: str

@dataclass
class Chunk:
    chunk_id: str
    text: str
    article_numbers: List[int] = field(default_factory=list)
    effective_mode: str = None
    file_name: str = None
    actors: List[str] = field(default_factory=list)