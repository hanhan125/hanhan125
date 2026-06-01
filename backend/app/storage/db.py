from collections.abc import Iterator

from sqlmodel import Session, SQLModel, create_engine, select

from app.core.config import settings
from app.storage.models import Classroom


def _sqlite_url() -> str:
    # Keep it file-based for stable local demo.
    return f"sqlite:///{settings.sqlite_path}"


engine = create_engine(_sqlite_url(), echo=False, connect_args={"check_same_thread": False})


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        if not session.exec(select(Classroom)).first():
            session.add(Classroom(name="默认教室"))
            session.commit()


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session

