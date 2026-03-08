from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional


class NewspaperJob(SQLModel, table=True):

    id: Optional[int] = Field(default=None, primary_key=True)

    image_path: str

    script_path: Optional[str] = None
    audio_path: Optional[str] = None

    articles_found: Optional[int] = None

    status: str = "uploaded"

    created_at: datetime = Field(default_factory=datetime.utcnow)

class User(SQLModel, table=True):

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    email: str
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)