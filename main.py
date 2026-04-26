import os
import uuid
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from sqlalchemy import Column, DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from starlette.middleware.sessions import SessionMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Config ────────────────────────────────────────────────────────────────────

SECRET_KEY    = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")
ADMIN_EMAIL   = os.getenv("ADMIN_EMAIL", "ilyas.kadri@gmail.com")
ADMIN_NAME    = os.getenv("ADMIN_NAME", "Ilyas")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "SentimentChat2024!")

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


_seed_admin()

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


# ── WebSocket ─────────────────────────────────────────────────────────────────

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
