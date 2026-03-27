"""
DenseLess — FastAPI Application Entry Point
=============================================
Sets up the FastAPI app, mounts static files,
configures Jinja2 templating, and includes routers.
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# ── Load environment variables from .env ─────────────────────────
load_dotenv()

# ── Resolve project paths ────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# ── Create the FastAPI application ───────────────────────────────
app = FastAPI(
    title="DenseLess",
    description="A student-centric Generative AI learning platform.",
    version="0.1.0",
)

# ── Mount static files (CSS, JS, images) ────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Set up Jinja2 template engine ────────────────────────────────
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Health-check / landing route ─────────────────────────────────
@app.get("/", tags=["General"])
async def index(request: Request):
    """Render the landing page (base template for now)."""
    return templates.TemplateResponse("base.html", {"request": request})
