# DenseLess

> A student-centric, web-based Generative AI learning platform designed to improve comprehension and long-term retention through Cognitive Load Adaptation and Spaced Repetition.
I used this project for generating the eval dataset

## Tech Stack

| Layer              | Technology                     |
|--------------------|--------------------------------|
| Frontend           | HTML5, CSS3, Jinja2, Vanilla JS |
| Backend            | Python, FastAPI                |
| AI Orchestration   | LangChain (Gemini / Ollama)    |
| Database & Auth    | Supabase (PostgreSQL)          |
| Hosting            | Vercel                         |

I enabled pgvector for vector usage in my supabase

## Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fill in your .env file with real credentials

# 4. Run the development server
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Project Structure

```
DenseLess/
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── database.py      # Supabase client
│   ├── routers/         # API route modules
│   ├── schemas/         # Pydantic models
│   └── agent/           # LangChain AI orchestration
├── templates/           # Jinja2 HTML templates
├── static/              # CSS, JS, images
├── .env                 # Environment variables (not committed)
├── requirements.txt
├── vercel.json
└── README.md
```
