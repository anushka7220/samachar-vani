from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select
from pathlib import Path
import shutil

from app.database import get_session, create_db_and_tables
from app.crud import create_job, get_job, update_job
from app.models import NewspaperJob, User
from pipeline.orchestrator import NewspaperPodcastPipeline
from app.auth import hash_password, verify_password
from fastapi import Form
from sqlmodel import select
from app.models import User
from app.auth import hash_password, verify_password
from groq import Groq
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# ---------------- APP ----------------

app = FastAPI(title="Hindi Newspaper Podcast API")

# serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ---------------- PATHS ----------------

UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


# ---------------- STARTUP ----------------

@app.on_event("startup")
def startup():
    create_db_and_tables()


# ---------------- PIPELINE RUNNER ----------------

def run_pipeline(job_id: int, image_path: str):

    session = next(get_session())

    job = get_job(session, job_id)

    job.status = "processing"
    update_job(session, job)

    pipeline = NewspaperPodcastPipeline(
        output_dir=f"results/job_{job_id}",
        top_n=3,
        tts_engine="edge-tts"
    )

    result = pipeline.run(image_path)

    job.status = "completed"
    job.script_path = result["script_path"]
    job.audio_path = result["audio_path"]
    job.articles_found = len(result["top_articles"])

    update_job(session, job)


# ---------------- UPLOAD ----------------

@app.post("/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
):

    image_path = UPLOAD_DIR / file.filename

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job = create_job(session, str(image_path))

    background_tasks.add_task(
        run_pipeline,
        job.id,
        str(image_path)
    )

    return {
        "job_id": job.id,
        "status": "uploaded"
    }


# ---------------- JOB STATUS ----------------

@app.get("/job/{job_id}")
def get_job_status(job_id: int, session: Session = Depends(get_session)):

    job = get_job(session, job_id)

    if not job:
        return {"error": "job not found"}

    return job


@app.get("/jobs")
def list_jobs(session: Session = Depends(get_session)):

    jobs = session.exec(select(NewspaperJob)).all()

    return jobs


# ---------------- AUDIO ----------------

@app.get("/audio/{job_id}")
def get_audio(job_id: int, session: Session = Depends(get_session)):

    job = get_job(session, job_id)

    if not job:
        return {"error": "job not found"}

    if not job.audio_path:
        return {"error": "audio not ready yet"}

    return FileResponse(
        job.audio_path,
        media_type="audio/mpeg",
        filename=f"podcast_{job_id}.mp3"
    )


# ---------------- SCRIPT ----------------

@app.get("/script/{job_id}")
def get_script(job_id: int, session: Session = Depends(get_session)):

    job = get_job(session, job_id)

    if not job:
        return {"error": "job not found"}

    if not job.script_path:
        return {"error": "script not ready yet"}

    return FileResponse(
        job.script_path,
        media_type="text/plain",
        filename=f"script_{job_id}.txt"
    )


# ---------------- AUTH ----------------

@app.post("/signup")
def signup(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session)
):

    existing = session.exec(
        select(User).where(User.email == email)
    ).first()

    if existing:
        return {"error": "email already registered"}

    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password)
    )

    session.add(user)
    session.commit()
    session.refresh(user)

    return {"message": "user created", "user_id": user.id}


@app.post("/login")
def login(
    email: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session)
):

    user = session.exec(
        select(User).where(User.email == email)
    ).first()

    if not user:
        return {"error": "invalid credentials"}

    if not verify_password(password, user.password_hash):
        return {"error": "invalid credentials"}

    return {
        "message": "login success",
        "user_id": user.id,
        "username": user.username,
        "email": user.email
    }
# ---------------- FRONTEND ----------------

@app.get("/", response_class=HTMLResponse)
def homepage():

    with open("frontend/index.html") as f:
        return f.read()