import json
import os
import uuid
from datetime import datetime
from typing import Dict, List

from anthropic import AsyncAnthropic
from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from sqlalchemy import Column, DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from starlette.middleware.sessions import SessionMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Config ────────────────────────────────────────────────────────────────────

SECRET_KEY       = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")
ADMIN_EMAIL      = os.getenv("ADMIN_EMAIL", "ilyas.kadri@gmail.com")
ADMIN_NAME       = os.getenv("ADMIN_NAME", "Ilyas")
ADMIN_PASSWORD   = os.getenv("ADMIN_PASSWORD", "SentimentChat2024!")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Database ──────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id            = Column(String, primary_key=True)
    email         = Column(String, unique=True, index=True, nullable=False)
    name          = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    id              = Column(String, primary_key=True)
    room            = Column(String, index=True, nullable=False)
    username        = Column(String, nullable=False)
    content         = Column(Text, nullable=False)
    sentiment_label = Column(String, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow)


class CareerProfile(Base):
    __tablename__ = "career_profile"
    id           = Column(String, primary_key=True)
    user_id      = Column(String, index=True, nullable=False, unique=True)
    full_name    = Column(String, default="")
    email        = Column(String, default="")
    headline     = Column(String, default="")
    skills       = Column(Text, default="[]")
    experience   = Column(Text, default="[]")
    target_roles = Column(Text, default="[]")
    cv_text      = Column(Text, default="")
    notes        = Column(Text, default="")
    updated_at   = Column(DateTime, default=datetime.utcnow)


class JobApplication(Base):
    __tablename__ = "job_applications"
    id           = Column(String, primary_key=True)
    user_id      = Column(String, index=True, nullable=False)
    company      = Column(String, nullable=False)
    role         = Column(String, nullable=False)
    status       = Column(String, default="in-progress")
    cover_letter = Column(Text, default="")
    notes        = Column(Text, default="")
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ── Seed admin user on first boot ─────────────────────────────────────────────

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _seed_admin():
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.email == ADMIN_EMAIL).first():
            db.add(User(
                id=str(uuid.uuid4()),
                email=ADMIN_EMAIL,
                name=ADMIN_NAME,
                password_hash=pwd_ctx.hash(ADMIN_PASSWORD),
            ))
            db.commit()
    finally:
        db.close()


def _seed_career_profile():
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.email == ADMIN_EMAIL).first()
        if not admin:
            return
        if db.query(CareerProfile).filter(CareerProfile.user_id == admin.id).first():
            return
        db.add(CareerProfile(
            id=str(uuid.uuid4()),
            user_id=admin.id,
            full_name="Ilyas Kadri",
            email="ilyas.kadri@gmail.com",
            headline="Software Engineer · AI & Real-time Systems",
            skills=json.dumps([
                "Python", "FastAPI", "WebSockets", "SQLAlchemy",
                "AI/ML", "Sentiment Analysis (VADER/NLP)",
                "Tailwind CSS", "PostgreSQL", "Railway", "Docker",
                "Real-time Systems", "REST APIs",
            ]),
            experience=json.dumps([
                {
                    "company": "Personal / Open-source",
                    "role": "Full-stack Developer",
                    "description": (
                        "Built SentimentChat — a production-grade real-time chat platform "
                        "with live AI sentiment analysis per message, WebSocket broadcasting, "
                        "multi-room support, analytics dashboard, and Railway deployment."
                    ),
                }
            ]),
            target_roles=json.dumps([
                "Software Engineer", "AI Engineer",
                "Backend Developer", "Full-stack Developer",
            ]),
            notes=(
                "Strong hands-on builder. Comfortable taking an idea from zero to deployed product. "
                "Interested in roles at the intersection of AI and product engineering."
            ),
        ))
        db.commit()
    finally:
        db.close()


_seed_admin()
_seed_career_profile()

# ── Sentiment ─────────────────────────────────────────────────────────────────

_analyzer = SentimentIntensityAnalyzer()


def analyze(text: str) -> dict:
    compound = round(_analyzer.polarity_scores(text)["compound"], 3)
    if compound >= 0.05:
        label, emoji = "positive", "😊"
    elif compound <= -0.05:
        label, emoji = "negative", "😟"
    else:
        label, emoji = "neutral", "😐"
    return {"label": label, "score": compound, "emoji": emoji}


# ── Career AI helpers ──────────────────────────────────────────────────────────

CAREER_TOOLS = [
    {
        "name": "save_job_application",
        "description": (
            "Save or update a job application in the tracker. Call this whenever you help "
            "the user with a specific company and role — writing a cover letter, doing "
            "interview prep, or when they mention they applied somewhere."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company":      {"type": "string", "description": "Company name"},
                "role":         {"type": "string", "description": "Job title / role"},
                "status":       {
                    "type": "string",
                    "enum": ["in-progress", "applied", "interview", "offer", "rejected"],
                    "description": "Current status of the application",
                },
                "cover_letter": {"type": "string", "description": "Cover letter if one was written"},
                "notes":        {"type": "string", "description": "Key talking points or action items"},
            },
            "required": ["company", "role", "status"],
        },
    },
    {
        "name": "update_profile",
        "description": "Update the user's career profile when they share new info about skills, experience, or goals.",
        "input_schema": {
            "type": "object",
            "properties": {
                "headline":     {"type": "string"},
                "skills":       {"type": "array", "items": {"type": "string"}},
                "experience":   {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "company":     {"type": "string"},
                            "role":        {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "target_roles": {"type": "array", "items": {"type": "string"}},
                "cv_text":      {"type": "string", "description": "Raw CV / resume text if shared"},
                "notes":        {"type": "string"},
            },
        },
    },
]


def _build_system_prompt(profile: CareerProfile | None, username: str) -> str:
    if profile:
        skills      = json.loads(profile.skills or "[]")
        experience  = json.loads(profile.experience or "[]")
        targets     = json.loads(profile.target_roles or "[]")
        exp_lines   = "\n".join(
            f"  • {e.get('role','?')} at {e.get('company','?')}: {e.get('description','')}"
            for e in experience
        ) or "  Not yet specified"
        profile_block = f"""Name: {profile.full_name or username}
Email: {profile.email}
Headline: {profile.headline or 'Software Professional'}
Skills: {', '.join(skills) or 'Not specified'}
Experience:
{exp_lines}
Target Roles: {', '.join(targets) or 'Not specified'}
{('CV/Resume on file: yes' if profile.cv_text else '')}
{('Notes: ' + profile.notes) if profile.notes else ''}"""
    else:
        profile_block = f"Name: {username}\n(Profile not yet populated)"

    return f"""You are JobCoach — a sharp, encouraging AI career assistant for {profile.full_name if profile else username}. \
You have their full profile loaded and your job is to help them land their next role.

## Candidate Profile
{profile_block}

## What you do
- Write compelling, tailored cover letters (always ask for the job description if not provided)
- Review and improve CV content, language, and positioning
- Run mock interviews with role-specific questions and give honest feedback
- Advise on job search strategy, LinkedIn optimisation, and salary negotiation
- Track every application automatically using your tools

## How you work
- Be specific — reference their actual skills and projects, not generic advice
- Use clear formatting: bullets, short paragraphs, headers where helpful
- When you help with a specific company + role, always call save_job_application to log it
- When the user shares new skills, experience, or goals, call update_profile
- If asked to write a cover letter, produce the full text ready to copy
- Today is {datetime.utcnow().strftime('%B %d, %Y')}
"""


async def _handle_tool(tool_name: str, tool_input: dict, user_id: str) -> dict:
    db = SessionLocal()
    try:
        if tool_name == "save_job_application":
            company      = tool_input.get("company", "")
            role         = tool_input.get("role", "")
            status       = tool_input.get("status", "in-progress")
            cover_letter = tool_input.get("cover_letter", "")
            notes        = tool_input.get("notes", "")

            existing = (
                db.query(JobApplication)
                .filter(JobApplication.user_id == user_id, JobApplication.company == company, JobApplication.role == role)
                .first()
            )
            if existing:
                if status:       existing.status       = status
                if cover_letter: existing.cover_letter = cover_letter
                if notes:        existing.notes        = notes
                existing.updated_at = datetime.utcnow()
                db.commit()
                return {"action": "updated", "id": existing.id, "company": company, "role": role, "status": status}
            else:
                app = JobApplication(
                    id=str(uuid.uuid4()), user_id=user_id,
                    company=company, role=role, status=status,
                    cover_letter=cover_letter, notes=notes,
                )
                db.add(app)
                db.commit()
                return {"action": "created", "id": app.id, "company": company, "role": role, "status": status}

        elif tool_name == "update_profile":
            profile = db.query(CareerProfile).filter(CareerProfile.user_id == user_id).first()
            if profile:
                if "headline"     in tool_input: profile.headline     = tool_input["headline"]
                if "skills"       in tool_input: profile.skills       = json.dumps(tool_input["skills"])
                if "experience"   in tool_input: profile.experience   = json.dumps(tool_input["experience"])
                if "target_roles" in tool_input: profile.target_roles = json.dumps(tool_input["target_roles"])
                if "cv_text"      in tool_input: profile.cv_text      = tool_input["cv_text"]
                if "notes"        in tool_input: profile.notes        = tool_input["notes"]
                profile.updated_at = datetime.utcnow()
                db.commit()
            return {"action": "profile_updated"}

        return {"action": "unknown_tool"}
    finally:
        db.close()


# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.rooms: Dict[str, List[WebSocket]] = {}

    async def connect(self, ws: WebSocket, room: str):
        await ws.accept()
        self.rooms.setdefault(room, []).append(ws)

    def disconnect(self, ws: WebSocket, room: str):
        if room in self.rooms:
            self.rooms[room] = [c for c in self.rooms[room] if c is not ws]

    async def broadcast(self, data: dict, room: str):
        for ws in list(self.rooms.get(room, [])):
            try:
                await ws.send_json(data)
            except Exception:
                pass

    def count(self, room: str) -> int:
        return len(self.rooms.get(room, []))


manager = ConnectionManager()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="SentimentChat")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def _session_user(request: Request) -> dict | None:
    uid = request.session.get("user_id")
    if not uid:
        return None
    return {
        "id":    uid,
        "name":  request.session.get("user_name"),
        "email": request.session.get("user_email"),
    }


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    if _session_user(request):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email.lower().strip()).first()
        if not user or not pwd_ctx.verify(password, user.password_hash):
            return templates.TemplateResponse("login.html", {
                "request": request,
                "error": "Invalid email or password.",
            })
        request.session["user_id"]    = user.id
        request.session["user_name"]  = user.name
        request.session["user_email"] = user.email
        return RedirectResponse("/", status_code=303)
    finally:
        db.close()


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = _session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


@app.get("/chat/{room}", response_class=HTMLResponse)
async def chat(request: Request, room: str):
    user = _session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "room": room,
        "username": user["name"],
    })


@app.get("/dashboard/{room}", response_class=HTMLResponse)
async def dashboard(request: Request, room: str):
    user = _session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "room": room,
        "user": user,
    })


@app.get("/career", response_class=HTMLResponse)
async def career_page(request: Request):
    user = _session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    db = SessionLocal()
    try:
        profile = db.query(CareerProfile).filter(CareerProfile.user_id == user["id"]).first()
        profile_data = {
            "full_name":    profile.full_name if profile else user["name"],
            "headline":     profile.headline  if profile else "",
            "skills":       json.loads(profile.skills or "[]") if profile else [],
            "target_roles": json.loads(profile.target_roles or "[]") if profile else [],
        } if profile else {}
    finally:
        db.close()
    return templates.TemplateResponse("career.html", {
        "request": request,
        "user": user,
        "profile": profile_data,
        "has_api_key": bool(ANTHROPIC_API_KEY),
    })


@app.get("/career/applications", response_class=HTMLResponse)
async def applications_page(request: Request):
    user = _session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("applications.html", {"request": request, "user": user})


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/api/messages/{room}")
def get_messages(room: str, limit: int = 50):
    db = SessionLocal()
    try:
        rows = (
            db.query(Message)
            .filter(Message.room == room)
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id":              m.id,
                "username":        m.username,
                "content":         m.content,
                "sentiment_label": m.sentiment_label,
                "sentiment_score": m.sentiment_score,
                "created_at":      m.created_at.isoformat(),
            }
            for m in reversed(rows)
        ]
    finally:
        db.close()


@app.get("/api/analytics/{room}")
def get_analytics(room: str):
    db = SessionLocal()
    try:
        msgs  = db.query(Message).filter(Message.room == room).all()
        total = len(msgs)
        if not total:
            return {
                "total": 0, "positive": 0, "negative": 0, "neutral": 0,
                "avg_score": 0, "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0,
            }
        pos = sum(1 for m in msgs if m.sentiment_label == "positive")
        neg = sum(1 for m in msgs if m.sentiment_label == "negative")
        neu = total - pos - neg
        avg = round(sum(m.sentiment_score for m in msgs) / total, 3)
        return {
            "total":        total,
            "positive":     pos,
            "negative":     neg,
            "neutral":      neu,
            "avg_score":    avg,
            "positive_pct": round(pos / total * 100, 1),
            "negative_pct": round(neg / total * 100, 1),
            "neutral_pct":  round(neu / total * 100, 1),
        }
    finally:
        db.close()


@app.get("/api/career/profile")
def get_career_profile(request: Request):
    user = _session_user(request)
    if not user:
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    db = SessionLocal()
    try:
        p = db.query(CareerProfile).filter(CareerProfile.user_id == user["id"]).first()
        if not p:
            return {}
        return {
            "full_name":    p.full_name,
            "email":        p.email,
            "headline":     p.headline,
            "skills":       json.loads(p.skills or "[]"),
            "experience":   json.loads(p.experience or "[]"),
            "target_roles": json.loads(p.target_roles or "[]"),
            "cv_text":      p.cv_text,
            "notes":        p.notes,
            "updated_at":   p.updated_at.isoformat() if p.updated_at else None,
        }
    finally:
        db.close()


@app.patch("/api/career/profile")
async def patch_career_profile(request: Request):
    user = _session_user(request)
    if not user:
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    body = await request.json()
    db = SessionLocal()
    try:
        p = db.query(CareerProfile).filter(CareerProfile.user_id == user["id"]).first()
        if not p:
            return JSONResponse({"error": "profile not found"}, status_code=404)
        for field in ("headline", "skills", "experience", "target_roles", "cv_text", "notes"):
            if field in body:
                val = body[field]
                setattr(p, field, json.dumps(val) if isinstance(val, (list, dict)) else val)
        p.updated_at = datetime.utcnow()
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.get("/api/career/applications")
def get_applications(request: Request):
    user = _session_user(request)
    if not user:
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    db = SessionLocal()
    try:
        apps = (
            db.query(JobApplication)
            .filter(JobApplication.user_id == user["id"])
            .order_by(JobApplication.updated_at.desc())
            .all()
        )
        return [
            {
                "id":           a.id,
                "company":      a.company,
                "role":         a.role,
                "status":       a.status,
                "cover_letter": a.cover_letter,
                "notes":        a.notes,
                "created_at":   a.created_at.isoformat(),
                "updated_at":   a.updated_at.isoformat(),
            }
            for a in apps
        ]
    finally:
        db.close()


@app.patch("/api/career/applications/{app_id}")
async def update_application(app_id: str, request: Request):
    user = _session_user(request)
    if not user:
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    body = await request.json()
    db = SessionLocal()
    try:
        a = db.query(JobApplication).filter(
            JobApplication.id == app_id, JobApplication.user_id == user["id"]
        ).first()
        if not a:
            return JSONResponse({"error": "not found"}, status_code=404)
        if "status" in body: a.status = body["status"]
        if "notes"  in body: a.notes  = body["notes"]
        a.updated_at = datetime.utcnow()
        db.commit()
        return {"ok": True}
    finally:
        db.close()


# ── WebSocket: job coach AI — must be defined BEFORE the generic /ws/{room}/{username}
#    so Starlette matches /ws/career/* here instead of falling into the group chat handler

@app.websocket("/ws/career/{user_id}")
async def career_ws(ws: WebSocket, user_id: str):
    await ws.accept()

    if not ANTHROPIC_API_KEY:
        await ws.send_json({"type": "error", "message": "ANTHROPIC_API_KEY is not configured on this server."})
        await ws.close()
        return

    db = SessionLocal()
    try:
        profile = db.query(CareerProfile).filter(CareerProfile.user_id == user_id).first()
        user    = db.query(User).filter(User.id == user_id).first()
    finally:
        db.close()

    system_prompt = _build_system_prompt(profile, user.name if user else user_id)
    history: list[dict] = []
    ai = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    try:
        while True:
            text = (await ws.receive_text()).strip()
            if not text:
                continue

            history.append({"role": "user", "content": text})
            await ws.send_json({"type": "thinking"})

            # Agentic loop — keep going until Claude stops using tools
            while True:
                response = await ai.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=2048,
                    system=system_prompt,
                    messages=history,
                    tools=CAREER_TOOLS,
                )

                reply_text = ""
                tool_calls = []
                for block in response.content:
                    if block.type == "text":
                        reply_text += block.text
                    elif block.type == "tool_use":
                        tool_calls.append(block)

                history.append({"role": "assistant", "content": response.content})

                if not tool_calls:
                    await ws.send_json({"type": "ai_response", "content": reply_text})
                    break

                # Execute tools and feed results back
                tool_results = []
                for tc in tool_calls:
                    result = await _handle_tool(tc.name, tc.input, user_id)
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": tc.id,
                        "content":     json.dumps(result),
                    })
                    await ws.send_json({"type": "tool_action", "tool": tc.name, "data": result})

                history.append({"role": "user", "content": tool_results})

    except WebSocketDisconnect:
        pass


# ── WebSocket: group chat ─────────────────────────────────────────────────────

@app.websocket("/ws/{room}/{username}")
async def ws_endpoint(ws: WebSocket, room: str, username: str):
    await manager.connect(ws, room)
    await manager.broadcast(
        {"type": "system", "message": f"{username} joined the room", "users": manager.count(room)},
        room,
    )
    try:
        while True:
            text = (await ws.receive_text()).strip()
            if not text:
                continue

            sentiment = analyze(text)
            db = SessionLocal()
            try:
                msg = Message(
                    id=str(uuid.uuid4()),
                    room=room,
                    username=username,
                    content=text,
                    sentiment_label=sentiment["label"],
                    sentiment_score=sentiment["score"],
                )
                db.add(msg)
                db.commit()
                payload = {
                    "type":       "message",
                    "id":         msg.id,
                    "username":   username,
                    "content":    text,
                    "sentiment":  sentiment,
                    "created_at": msg.created_at.isoformat(),
                }
            finally:
                db.close()

            await manager.broadcast(payload, room)

    except WebSocketDisconnect:
        manager.disconnect(ws, room)
        await manager.broadcast(
            {"type": "system", "message": f"{username} left the room", "users": manager.count(room)},
            room,
        )
