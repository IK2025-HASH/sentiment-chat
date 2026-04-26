import os
import uuid
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Column, DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Database ──────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat.db")
# Railway exposes postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True)
    room = Column(String, index=True, nullable=False)
    username = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sentiment_label = Column(String, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

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
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat/{room}", response_class=HTMLResponse)
async def chat(request: Request, room: str):
    return templates.TemplateResponse("chat.html", {"request": request, "room": room})


@app.get("/dashboard/{room}", response_class=HTMLResponse)
async def dashboard(request: Request, room: str):
    return templates.TemplateResponse("dashboard.html", {"request": request, "room": room})


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
                "id": m.id,
                "username": m.username,
                "content": m.content,
                "sentiment_label": m.sentiment_label,
                "sentiment_score": m.sentiment_score,
                "created_at": m.created_at.isoformat(),
            }
            for m in reversed(rows)
        ]
    finally:
        db.close()


@app.get("/api/analytics/{room}")
def get_analytics(room: str):
    db = SessionLocal()
    try:
        msgs = db.query(Message).filter(Message.room == room).all()
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
            "total": total,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "avg_score": avg,
            "positive_pct": round(pos / total * 100, 1),
            "negative_pct": round(neg / total * 100, 1),
            "neutral_pct": round(neu / total * 100, 1),
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
                    "type": "message",
                    "id": msg.id,
                    "username": username,
                    "content": text,
                    "sentiment": sentiment,
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
