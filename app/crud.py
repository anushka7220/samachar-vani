from sqlmodel import Session, select
from .models import NewspaperJob


def create_job(session: Session, image_path: str):

    job = NewspaperJob(
        image_path=image_path,
        status="uploaded"
    )

    session.add(job)
    session.commit()
    session.refresh(job)

    return job


def get_job(session: Session, job_id: int):

    statement = select(NewspaperJob).where(NewspaperJob.id == job_id)

    return session.exec(statement).first()


def update_job(session: Session, job: NewspaperJob):

    session.add(job)
    session.commit()
    session.refresh(job)

    return job