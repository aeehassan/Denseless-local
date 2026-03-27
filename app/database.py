"""
DenseLess — Supabase Database Client
======================================
Initialises a Supabase client using credentials
loaded from environment variables.
"""

import os
from supabase import create_client, Client

# ── Read credentials from .env ───────────────────────────────────
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

# ── Validate that credentials are present ────────────────────────
if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError(
        "Missing SUPABASE_URL or SUPABASE_KEY in your .env file. "
        "Please set them before starting the server."
    )

# ── Create and export the Supabase client ────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
