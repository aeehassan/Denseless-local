"""
Denseless — Streamlit interface
Entry point: handles login / new-user creation, then routes to the
sidebar-navigated app (dashboard, library, notes, quiz, analytics).

Run with:
    streamlit run frontend.py

Note that: There are provisions below to select the llm you want the
app to use by accessing the llm variable then uncommenting and filling
the right parameter values.
"""

import shutil
import json, os
import base64
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from collections import defaultdict
from datetime import datetime, timedelta

from app.agent.rag.ingestion.data_ingestion import process_and_load_file
from app.agent.rag.ingestion.embeddings import get_embedding_model, generate_embeddings
from app.agent.rag.chains.qa_chain import run_qa_chain
from app.agent.rag.ingestion.vector_store import (
    get_vector_store,
    add_documents_to_chroma,
)
from app.services.quiz_router import handle_quiz_request
from app.agent.rag.chains.eval_chain import run_eval_chain
from app.agent.rag.retrieval.retriever import get_topic_chunks
from app.agent.rag.chains.notes_chain import run_notes_chain

from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# --------------------------------------------------------------------------
# Config — adjust these two paths if your project layout differs
# --------------------------------------------------------------------------
PROFILES_DIR = Path("data/profiles")  # mirrors data/quizzes, data/notes
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

PDF_DIR = Path("app/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

AI_PDF_DIR = Path("data/notes")
AI_PDF_DIR.mkdir(parents=True, exist_ok=True)

CHAT_DIR = Path("data/chat_history")
CHAT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_PASSWORD = "pass"
ID_PATTERN = re.compile(r"^\d{4}$")


def profile_path(student_number: str) -> Path:
    """student_number is the raw 4-digit id, e.g. '1019' -> student_1019.json"""
    return PROFILES_DIR / f"student_{student_number}.json"


def blank_profile(student_id: str, pace: str) -> dict:
    """Fresh profile scaffold matching the existing schema."""
    return {
        "user_id": student_id,
        "learning_pace": pace,
        "tokens_remaining": 10_000_000,
        "tokens_used": 0,
        "token_history": [],
        "weak_topics": [],
        "strong_topics": [],
        "topics": {},
        "scores": {"comprehension": {}, "retention": {}},
        "revision_dates": {},
    }


def load_profile(student_number: str) -> dict:
    with open(profile_path(student_number), "r", encoding="utf-8") as f:
        return json.load(f)


def save_profile(student_number: str, profile: dict) -> None:
    with open(profile_path(student_number), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def profile_exists(student_number: str) -> bool:
    return profile_path(student_number).exists()


def render_pdf(file_path: str):
    """Encodes a local PDF file and renders it inside an HTML iframe."""

    if not file_path or not Path(file_path).exists():
        st.error(f"Cannot find PDF at path: {file_path}")
        return

    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        # Embedding PDF in HTML with a fixed height to match the workspace layout
        pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="750px" 
            type="application/pdf"
            style="border: none; border-radius: 5px;">
        </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading PDF: {e}")


def get_chat_memory_path(student_id: str, chat_name: str) -> Path:
    """Returns the exact path to the summary memory JSON."""
    return CHAT_DIR / student_id / f"{chat_name}.json"


def load_chat_memory(student_id: str, chat_name: str) -> dict:
    path = get_chat_memory_path(student_id, chat_name)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"summary": "No memory established yet.", "turn_count": 0}


# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "student_number" not in st.session_state:
    st.session_state.student_number = None
if "student_id" not in st.session_state:
    st.session_state.student_id = None  # e.g. "student_1019"


# --------------------------------------------------------------------------
# Login screen
# --------------------------------------------------------------------------
def render_login():
    st.set_page_config(
        page_title="Denseless — Login", page_icon="📘", layout="centered"
    )
    st.title("📘 Denseless")
    st.caption(
        "Adaptive RAG-based tutoring system focused on comprehension and retention"
    )

    tab_login, tab_new = st.tabs(["Log in", "Create new profile"])

    with tab_login:
        st.subheader("Log in")
        id_input = st.text_input("Student ID (4 digits)", max_chars=4, key="login_id")
        pw_input = st.text_input("Password", type="password", key="login_pw")

        if st.button("Log in", type="primary", use_container_width=True):
            if not ID_PATTERN.match(id_input or ""):
                st.error("ID must be exactly 4 digits.")
            elif pw_input != FIXED_PASSWORD:
                st.error("Incorrect password.")
            elif not profile_exists(id_input):
                st.error(
                    "No profile found for this ID. Use 'Create new profile' instead."
                )
            else:
                st.session_state.authenticated = True
                st.session_state.student_number = id_input
                st.session_state.student_id = f"student_{id_input}"

                with st.spinner("Initializing your workspace..."):
                    # 1. Initialize LLM
                    if "llm" not in st.session_state:
                        # Using mistral:latest
                        # Instantiate your llm (local or paid)

                        # local:
                        # llm = ChatOllama(model="mistral:latest")
                        load_dotenv()
                        # api_key = os.environ.get('GOOGLE_API_KEY')
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash-lite",
                            project="denseless",
                            location="us-central1",
                            vertexai=True,
                        )

                        st.session_state.llm = llm

                    # 2. Initialize Embedder
                    if "embedder" not in st.session_state:
                        st.session_state.embedder = get_embedding_model()

                    # 3. Initialize Vector Store (using the raw 4-digit ID)
                    if "store" not in st.session_state:
                        st.session_state.store = get_vector_store(
                            student_id=id_input, embedder=st.session_state.embedder
                        )

                st.rerun()

    with tab_new:
        st.subheader("Create a new profile")
        st.caption("Pick any unused 4-digit ID. Password is always 'pass'.")
        # Student id
        new_id_input = st.text_input(
            "New Student ID (4 digits)", max_chars=4, key="new_id"
        )
        # Learner pace
        new_pace_input = st.text_input("Learner Pace", max_chars=7, key="pace")

        if st.button("Create profile", use_container_width=True):
            if not ID_PATTERN.match(new_id_input or ""):
                st.error("ID must be exactly 4 digits.")
            elif profile_exists(new_id_input):
                st.error(
                    "A profile with this ID already exists. Please log in instead."
                )
            else:
                new_profile = blank_profile(f"student_{new_id_input}", new_pace_input)
                save_profile(new_id_input, new_profile)
                st.success(
                    f"Profile 'student_{new_id_input}' created. You can log in now."
                )


# --------------------------------------------------------------------------
# Dashboard page
# --------------------------------------------------------------------------
def render_dashboard():
    # Load profile using the correct raw 4-digit ID
    profile = load_profile(st.session_state.student_number)

    if not profile:
        st.error("Failed to load student profile. Please log in again.")
        return

    # Header
    st.markdown("## Welcome Back, Abubakar")
    st.markdown("Here is your learning kit...")
    st.write("")

    # --- ROW 1: Pace & Tokens ---
    # Using height=260 locks both containers to the exact same vertical height
    col1, col2 = st.columns([1.5, 1])

    with col1:
        with st.container(border=True, height=260):
            pace = profile.get("learning_pace", "average").lower()
            st.markdown(f"### 🧭 Learning Pace")

            if pace == "fast":
                st.write(
                    "Your current pace is set to Fast. This favors broad topic coverage, generating high-level summaries and concise quizzes for rapid review."
                )
                slider_val = "Faster (Less Detail)"
            elif pace == "slow":
                st.write(
                    "Your current pace is set to Slow. This favors deep dive analysis, generating highly detailed summaries and comprehensive quizzes."
                )
                slider_val = "Slower (More Detail)"
            else:
                st.write(
                    "Your current pace is set to Average. This maintains a balance between deep dive analysis and broad topic coverage, generating moderately detailed summaries and mid-length quizzes."
                )
                slider_val = "Average"

            st.select_slider(
                "Pace Control",
                options=["Slower (More Detail)", "Average", "Faster (Less Detail)"],
                value=slider_val,
                disabled=True,
                label_visibility="collapsed",
            )

    with col2:
        with st.container(border=True, height=260):
            tokens_remaining = profile.get("tokens_remaining", 0)
            tokens_used = profile.get("tokens_used", 0)
            total = tokens_remaining + tokens_used

            st.markdown("### 🪙 Token Usage")
            st.caption("Current Plan Cycle")

            # Text forced to white
            st.markdown(
                f"<h2 style='text-align: center; color: white; margin-bottom: 0;'>{tokens_remaining:,}</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='text-align: center; font-size: 14px; margin-top: 0; color: white;'>Tokens Remaining</p>",
                unsafe_allow_html=True,
            )

            prog = tokens_used / total if total > 0 else 0.0
            st.progress(prog)

            c_used, c_tot = st.columns(2)
            c_used.markdown(
                f"<span style='font-size: 12px; color: white;'>Used: {tokens_used / 1000:.0f}k</span>",
                unsafe_allow_html=True,
            )
            c_tot.markdown(
                f"<div style='text-align: right; font-size: 12px; color: white;'>Total: {total / 1000000:.2f}M</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")

    # --- ROW 2: Revision Timeline ---
    st.markdown("### Revision Timeline")

    revisions = profile.get("revision_dates", {})
    pending_revs = []

    for topic_name, rev_list in revisions.items():
        for r in rev_list:
            if r.get("status") == "pending":
                pending_revs.append({"topic": topic_name, "date": r.get("date")})

    if pending_revs:
        # Sort chronologically (closest dates first)
        pending_revs.sort(key=lambda x: x["date"])

        # Sliced to 4 to ensure a maximum of 4 upcoming reviews are shown
        display_revs = pending_revs[:4]
        cols = st.columns(len(display_revs))

        for i, rev in enumerate(display_revs):
            with cols[i]:
                with st.container(border=True):
                    try:
                        d_obj = datetime.strptime(rev["date"], "%Y-%m-%d")
                        month = d_obj.strftime("%b").upper()
                        day = d_obj.strftime("%d")
                    except ValueError:
                        month = "TBD"
                        day = "--"

                    # Overflow controls added (white-space: nowrap, text-overflow: ellipsis)
                    # Adjusted padding and icon background to handle white text smoothly
                    st.markdown(
                        f"""
                    <div style="display: flex; justify-content: center; align-items: center; gap: 12px; height: 100%; padding: 10px 0;">
                        <div style="background-color: #333333; padding: 8px 12px; border-radius: 6px; text-align: center; min-width: 65px; display: flex; flex-direction: column; justify-content: center;">
                            <div style="font-size: 11px; color: white; font-weight: bold;">{month}</div>
                            <div style="font-size: 18px; color: white; font-weight: bold; line-height: 1.1; margin-top: 2px;">{day}</div>
                        </div>
                        <div style="overflow: hidden; display: flex; flex-direction: column; justify-content: center;">
                            <div style="font-weight: bold; font-size: 14px; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{rev["topic"]}">{rev["topic"]}</div>
                            <div style="font-size: 11px; color: #cccccc; margin-top: 2px;">Scheduled Review</div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
    else:
        with st.container(border=True):
            st.write("No upcoming revisions scheduled. Keep studying!")


# --------------------------------------------------------------------------
# Library page
# --------------------------------------------------------------------------
@st.dialog("Create New Topic")
def create_topic_modal():
    topic_name = st.text_input(
        "Topic Name", placeholder="e.g., Introduction to Algorithms"
    )

    uploaded_file = st.file_uploader("Upload Materials (PDF)", type=["pdf"])
    st.caption(
        "<span style='color: #d9534f; font-size: 12px;'>Maximum 40 pages per PDF</span>",
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("Create Topic", type="primary", use_container_width=True):
            if not topic_name:
                st.error("Please enter a topic name.")
            elif not uploaded_file:
                st.error("Please upload a PDF document.")
            else:
                with st.spinner(f"Ingesting '{topic_name}' into vector database..."):
                    # 1. Save file locally
                    safe_filename = (
                        f"{st.session_state.student_id}_{uploaded_file.name}"
                    )
                    file_path = PDF_DIR / safe_filename

                    if not file_path.exists():
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    # 2. Initialize Embedder and Store in Session State (to avoid reloading on reruns)
                    if "embedder" not in st.session_state:
                        st.session_state.embedder = get_embedding_model()

                    if "store" not in st.session_state:
                        st.session_state.store = get_vector_store(
                            student_id=st.session_state.student_number,
                            embedder=st.session_state.embedder,
                        )

                    # 3. Execute Ingestion Pipeline
                    docs = process_and_load_file(str(file_path))
                    vectors = generate_embeddings(docs, st.session_state.embedder)

                    add_documents_to_chroma(
                        store=st.session_state.store,
                        embeddings=vectors,
                        documents=docs,
                        condensed=False,
                        course="Default",  # Placeholder since courses are excluded
                        topic=topic_name,
                        file_name=uploaded_file.name,
                    )

                    # 4. Update the profile schema
                    profile = load_profile(st.session_state.student_number)
                    if "topics" not in profile:
                        profile["topics"] = {}

                    profile["topics"][topic_name] = {
                        "weak_areas": [],
                        "strong_areas": [],
                        "notes_reviewed": False,
                        "original_pdf_path": str(file_path),
                        "condensed_pdf_path": None,
                    }
                    save_profile(st.session_state.student_number, profile)

                    st.success("Topic successfully ingested!")
                    st.rerun()


def render_library():
    profile = load_profile(st.session_state.student_number)

    if not profile:
        st.error("Failed to load student profile. Please log in again.")
        return

    # Header section
    st.markdown(
        "<div style='font-size: 13px; color: gray;'>Library &nbsp;❯&nbsp; Topics</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<h1 style='margin-top: -15px;'>Topics</h1>", unsafe_allow_html=True)
    st.write("")

    # Tabs mapping to "All", "Strong", "Weak"
    tab_all, tab_strong, tab_weak = st.tabs(["All", "Strong", "Weak"])

    topics = profile.get("topics", {})
    weak_topics = profile.get("weak_topics", [])
    strong_topics = profile.get("strong_topics", [])

    def render_topic_list(topic_dict, filter_list=None):
        if not topic_dict:
            st.info("No topics available. Create one below to get started.")
            return

        count = 0
        for t_name in list(topic_dict.keys()):
            t_data = topic_dict[t_name]

            if filter_list is not None and t_name not in filter_list:
                continue

            count += 1
            with st.container(border=True):
                c1, c2 = st.columns([3.2, 1])

                with c1:
                    st.markdown(
                        f"<div style='font-weight: bold; font-size: 16px; margin-bottom: 5px; margin-top: 12px;'>{t_name}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<div style='color: gray; font-size: 13px;'>Course material, extracted concepts, and performance tracking.</div>",
                        unsafe_allow_html=True,
                    )

                with c2:
                    spacer, pop_col = st.columns([3, 1])
                    with pop_col:
                        with st.popover("⋮"):
                            st.write("Danger Zone")
                            if st.button(
                                "Delete Topic",
                                key=f"del_{t_name}_{filter_list}",
                                type="primary",
                                use_container_width=True,
                            ):
                                # Scrub from all profile dictionaries
                                profile["topics"].pop(t_name, None)
                                if t_name in profile.get("weak_topics", []):
                                    profile["weak_topics"].remove(t_name)
                                if t_name in profile.get("strong_topics", []):
                                    profile["strong_topics"].remove(t_name)
                                profile.get("scores", {}).get("comprehension", {}).pop(
                                    t_name, None
                                )
                                profile.get("scores", {}).get("retention", {}).pop(
                                    t_name, None
                                )
                                profile.get("revision_dates", {}).pop(t_name, None)

                                save_profile(st.session_state.student_number, profile)
                                st.rerun()

                    # State toggle based strictly on the path's existence
                    notes_ready = bool(t_data.get("condensed_pdf_path"))

                    if notes_ready:
                        if st.button(
                            "✅ AI Notes &nbsp; ❯",
                            key=f"btn_{t_name}_{filter_list}",
                            use_container_width=True,
                        ):
                            # 1. Update the state handoff variables
                            st.session_state.workspace_topic = t_name
                            st.session_state.workspace_original = t_data.get(
                                "original_pdf_path"
                            )
                            st.session_state.workspace_condensed = t_data.get(
                                "condensed_pdf_path"
                            )

                            # 2. Provide user clearance
                            st.success(
                                f"Notes loaded! You may now proceed to the AI Workspace from the sidebar to study '{t_name}'."
                            )
                    else:
                        if st.button(
                            "◯ Generate Notes &nbsp; ❯",
                            key=f"btn_{t_name}_{filter_list}",
                            use_container_width=True,
                        ):
                            with st.spinner(
                                f"Generating notes for {t_name}... this may take a moment."
                            ):
                                # 1. Ensure LLM is in session state
                                if "llm" not in st.session_state:
                                    st.error(
                                        "LLM not initialized. Please log out and log back in."
                                    )
                                    st.stop()

                                # 2. Attempt to retrieve chunks
                                topic_chunks = get_topic_chunks(
                                    store=st.session_state.store,
                                    topic=t_name,
                                    course="Default",
                                )

                                # 3. JIT Ingestion Fallback
                                if not topic_chunks:
                                    st.toast(
                                        f"Embeddings missing for {t_name}. Running Just-In-Time ingestion..."
                                    )
                                    original_pdf = t_data.get("original_pdf_path")

                                    if original_pdf and Path(original_pdf).exists():
                                        # Execute ingestion pipeline
                                        docs = process_and_load_file(original_pdf)
                                        vectors = generate_embeddings(
                                            docs, st.session_state.embedder
                                        )

                                        add_documents_to_chroma(
                                            store=st.session_state.store,
                                            embeddings=vectors,
                                            documents=docs,
                                            condensed=False,
                                            course="Default",
                                            topic=t_name,
                                            file_name=Path(original_pdf).name,
                                        )

                                        # Re-fetch chunks now that they exist
                                        topic_chunks = get_topic_chunks(
                                            store=st.session_state.store,
                                            topic=t_name,
                                            course="Default",
                                        )
                                    else:
                                        # Legacy topic missing the file path
                                        st.error(
                                            f"Cannot generate notes: Source PDF for '{t_name}' not found. This is likely a legacy topic. Please delete and recreate it."
                                        )
                                        st.stop()

                                # 4. Run the notes chain
                                result = run_notes_chain(
                                    student_id=st.session_state.student_id,
                                    topic_map=topic_chunks,
                                    current_topic=t_name,
                                    weak_topics=profile.get("weak_topics", []),
                                    strong_topics=profile.get("strong_topics", []),
                                    course="Default",
                                    learning_pace=profile.get(
                                        "learning_pace", "average"
                                    ),
                                    llm=st.session_state.llm,
                                    embedder=st.session_state.embedder,
                                    store=st.session_state.store,
                                )

                                # 5. Enforce Relative Pathing
                                filename = Path(result.content).name
                                clean_relative_path = f"data/notes/{filename}"

                                # 6. Update Profile Schema
                                current_profile = load_profile(
                                    st.session_state.student_number
                                )
                                current_profile["topics"][t_name][
                                    "condensed_pdf_path"
                                ] = clean_relative_path

                                save_profile(
                                    st.session_state.student_number, current_profile
                                )
                                st.rerun()

        if count == 0:
            st.info("No topics found in this category.")

    # Render content within each tab
    with tab_all:
        render_topic_list(topics)
        st.write("")
        if st.button("⊕ Create New Topic", use_container_width=True):
            create_topic_modal()

    with tab_strong:
        render_topic_list(topics, strong_topics)
        st.write("")
        if st.button(
            "⊕ Create New Topic", key="btn_create_strong", use_container_width=True
        ):
            create_topic_modal()

    with tab_weak:
        render_topic_list(topics, weak_topics)
        st.write("")
        if st.button(
            "⊕ Create New Topic", key="btn_create_weak", use_container_width=True
        ):
            create_topic_modal()


@st.dialog("Full Memory Summary")
def read_more_modal(summary_text: str):
    st.write(summary_text)
    if st.button("Close", use_container_width=True):
        st.rerun()


# --------------------------------------------------------------------------
# Workspace tab
# --------------------------------------------------------------------------
def render_workspace():
    # 1. State Verification
    topic = st.session_state.get("workspace_topic")
    orig_path = st.session_state.get("workspace_original")
    cond_path = st.session_state.get("workspace_condensed")

    if not topic:
        st.info(
            "No topic loaded into the workspace. Please head to the Library and click '✅ AI Notes' to begin studying."
        )
        return

    # Header
    st.markdown(f"## {topic}")
    st.write("")

    # 2. Main Layout: Adjusted to [2.2, 1.0] for a much wider PDF viewer
    doc_col, chat_col = st.columns([2.2, 1.4], gap="medium")

    # --- LEFT HAND SIDE: PDF Viewer ---
    with doc_col:
        # Tabs placed directly without the extra columns for the button
        tab_orig, tab_cond = st.tabs(["Original PDF", "Condensed Note"])

        # Tab content
        with tab_orig:
            if orig_path:
                render_pdf(orig_path)
            else:
                st.warning("Original PDF path is missing from the profile.")

        with tab_cond:
            if cond_path:
                render_pdf(cond_path)
            else:
                st.warning(
                    "Condensed PDF path is missing. Did the generation complete successfully?"
                )

    # --- RIGHT HAND SIDE: Chat Placeholder ---
    with chat_col:
        # --- Session State Initialization for Chat ---
        if "chat_view" not in st.session_state:
            st.session_state.chat_view = "history"
        if "active_chat" not in st.session_state:
            st.session_state.active_chat = None
        if "latest_q" not in st.session_state:
            st.session_state.latest_q = None
        if "latest_a" not in st.session_state:
            st.session_state.latest_a = None

        # --- RIGHT HAND SIDE: AI Chat ---
        with chat_col:
            # Height tweaked here
            with st.container(border=True, height=600):
                st.markdown("### 🤖 AI Academic Assistant")
                st.caption("Denseless AI")
                st.divider()

                student_chat_dir = CHAT_DIR / st.session_state.student_id
                student_chat_dir.mkdir(exist_ok=True)

                # --- VIEW 1: History ---
                if st.session_state.chat_view == "history":
                    if st.button(
                        "➕ New Chat", type="primary", use_container_width=True
                    ):
                        st.session_state.chat_view = "new"
                        st.rerun()

                    st.write("")
                    st.markdown("**Your Conversations**")

                    # Glob the physical JSON files to get the list of conversations
                    chat_files = list(student_chat_dir.glob("*.json"))

                    if not chat_files:
                        st.info("No active conversations for this topic.")
                    else:
                        for cf in chat_files:
                            chat_name = cf.stem

                            cc1, cc2 = st.columns([4, 1])
                            with cc1:
                                if st.button(
                                    f"💬 {chat_name}",
                                    key=f"open_{chat_name}",
                                    use_container_width=True,
                                ):
                                    st.session_state.active_chat = chat_name
                                    st.session_state.chat_view = "active"
                                    st.session_state.latest_q = None
                                    st.session_state.latest_a = None
                                    st.rerun()
                            with cc2:
                                with st.popover("⋮"):
                                    if st.button(
                                        "Delete", key=f"del_{chat_name}", type="primary"
                                    ):
                                        cf.unlink()
                                        st.rerun()

                # --- VIEW 2: New Chat Setup ---
                elif st.session_state.chat_view == "new":
                    if st.button("🔙 Back", use_container_width=True):
                        st.session_state.chat_view = "history"
                        st.rerun()

                    st.write("")
                    new_chat_name = st.text_input(
                        "Enter a name for this chat:",
                        placeholder="e.g., Chapter 1 Review",
                    )

                    if st.button(
                        "Create & Start", type="primary", use_container_width=True
                    ):
                        if not new_chat_name:
                            st.error("Please enter a name.")
                        else:
                            st.session_state.active_chat = new_chat_name
                            st.session_state.chat_view = "active"
                            st.session_state.latest_q = None
                            st.session_state.latest_a = None
                            st.rerun()

                # --- VIEW 3: Active Chat ---
                elif st.session_state.chat_view == "active":
                    active_chat = st.session_state.active_chat

                    c_back, c_title = st.columns([1, 3])
                    with c_back:
                        if st.button("🔙 Back"):
                            st.session_state.chat_view = "history"
                            st.rerun()
                    with c_title:
                        st.markdown(f"**{active_chat}**")

                    st.divider()

                    # 1. Display Summary Memory
                    memory_data = load_chat_memory(
                        st.session_state.student_id, active_chat
                    )
                    full_summary = memory_data.get("summary", "")

                    st.markdown("**Context Memory**")
                    # Trim logic
                    char_limit = 120
                    if len(full_summary) > char_limit:
                        st.info(f"{full_summary[:char_limit]}...")
                        if st.button("Read more", key="read_more_btn"):
                            read_more_modal(full_summary)
                    else:
                        st.info(full_summary)

                    st.divider()

                    # 2. Display Latest Question and Answer
                    if st.session_state.latest_q and st.session_state.latest_a:
                        with st.chat_message("user"):
                            st.write(st.session_state.latest_q)

                        with st.chat_message("assistant"):
                            st.write(st.session_state.latest_a)
                    else:
                        st.write("*Ask a question below to begin.*")

            # Chat input is placed outside the height-restricted container so it pins to the bottom of the column
        if st.session_state.chat_view == "active":
            # The toggle widget capturing the boolean state
            from_condensed = st.toggle(
                "Answer from Condensed Notes",
                value=False,
                help="Toggle on to restrict the AI's search to your generated notes instead of the original PDF.",
            )

            prompt = st.chat_input(f"Ask about {topic}...")
            if prompt:
                st.session_state.latest_q = prompt

                # Execute QA Chain
                with st.spinner("Denseless AI is thinking..."):
                    # Load student profile to get pace
                    profile = load_profile(st.session_state.student_number)
                    pace = profile.get("learning_pace", "average")

                    # Updated to match the new run_qa_chain signature exactly
                    response = run_qa_chain(
                        student_id=st.session_state.student_id,
                        question=prompt,
                        convo_name=st.session_state.active_chat,
                        from_condensed_notes=from_condensed,
                        store=st.session_state.store,
                        llm=st.session_state.llm,
                        learning_pace=pace,
                        current_topic=topic,
                        course="Default",
                    )

                    # Extract the answer from the returned dictionary payload
                    st.session_state.latest_a = response.content.get(
                        "answer", "Error generating answer."
                    )
                    st.rerun()


def render_quiz():
    # --- Session State Initialization for Quiz ---
    if "quiz_view" not in st.session_state:
        st.session_state.quiz_view = "topics"
    if "quiz_topic" not in st.session_state:
        st.session_state.quiz_topic = None
    if "active_quiz_data" not in st.session_state:
        st.session_state.active_quiz_data = None
    if "view_quiz_path" not in st.session_state:
        st.session_state.view_quiz_path = None

    profile = load_profile(st.session_state.student_number)
    if not profile:
        st.error("Profile not found.")
        return

    # --- VIEW 1: Topic Selection ---
    if st.session_state.quiz_view == "topics":
        st.markdown(
            "<div style='font-size: 13px; color: gray;'>Quizzes &nbsp;❯&nbsp; Topics</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h1 style='margin-top: -15px;'>Select a Topic</h1>", unsafe_allow_html=True
        )
        st.write("")

        topics = profile.get("topics", {})
        if not topics:
            st.info(
                "No topics available. Head to the Library to ingest materials first."
            )
            return

        for t_name in topics.keys():
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(
                        f"<div style='font-weight: bold; font-size: 18px; margin-top: 8px;'>{t_name}</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button(
                        "View Quizzes ❯", key=f"vq_{t_name}", use_container_width=True
                    ):
                        st.session_state.quiz_topic = t_name
                        st.session_state.quiz_view = "detail"
                        st.rerun()

    # --- VIEW 2: Topic Detail & Triggers ---
    elif st.session_state.quiz_view == "detail":
        topic = st.session_state.quiz_topic

        c_back, c_title = st.columns([1, 4])
        with c_back:
            if st.button("🔙 Back"):
                st.session_state.quiz_view = "topics"
                st.rerun()
        with c_title:
            st.markdown(f"### {topic}")

        st.divider()

        # Main Layout
        hist_col, action_col = st.columns([2, 1], gap="large")

        with hist_col:
            st.markdown("#### Quiz History & Schedule")

            comp_scores = (
                profile.get("scores", {}).get("comprehension", {}).get(topic, [])
            )
            ret_scores = profile.get("scores", {}).get("retention", {}).get(topic, [])
            rev_dates = profile.get("revision_dates", {}).get(topic, [])

            if not comp_scores and not rev_dates:
                st.info("No quizzes taken or scheduled yet for this topic.")
            else:
                # ---------------------------------------------------------
                # 1. Render Comprehension Cards (Pre-Test & Post-Test)
                # ---------------------------------------------------------
                if comp_scores:
                    st.markdown("##### Initial Learning")
                    for idx, attempt in enumerate(comp_scores):
                        with st.container(border=True):
                            score_val = attempt.get("score")
                            score_str = (
                                f"{score_val}/10.0"
                                if score_val is not None
                                else "Pending"
                            )

                            phase_label = (
                                "Pre-Test"
                                if attempt.get("attempt", idx) == 0
                                else "Post-Test"
                            )

                            cc1, cc2 = st.columns([3, 1])
                            with cc1:
                                st.markdown(
                                    f"**Phase:** {phase_label} (Attempt {attempt.get('attempt', idx)})"
                                )
                                st.write(
                                    f"**Score:** {score_str} &nbsp;&nbsp;|&nbsp;&nbsp; **Date:** {attempt.get('date')}"
                                )
                            with cc2:
                                quiz_path = attempt.get("quiz_path")
                                if quiz_path:
                                    st.write("")
                                    if score_val is None:
                                        if st.button(
                                            "▶ Take Quiz",
                                            key=f"t_comp_{topic}_{idx}",
                                            type="primary",
                                            use_container_width=True,
                                        ):
                                            with open(
                                                Path(quiz_path), "r", encoding="utf-8"
                                            ) as f:
                                                quiz_data = json.load(f)

                                            st.session_state.active_quiz_data = {
                                                "phase": "pre_test"
                                                if attempt.get("attempt", idx) == 0
                                                else "post_test",
                                                "quiz": quiz_data,
                                                "saved_path": quiz_path,
                                            }
                                            # FIX: Carry over the exact date for evaluation
                                            st.session_state.simulated_date_str = (
                                                attempt.get("date")
                                            )
                                            st.session_state.quiz_view = "take"
                                            st.rerun()
                                    else:
                                        if st.button(
                                            "View Results",
                                            key=f"v_comp_{topic}_{idx}",
                                            use_container_width=True,
                                        ):
                                            st.session_state.view_quiz_path = quiz_path
                                            st.session_state.quiz_view = "results"
                                            st.rerun()

                # ---------------------------------------------------------
                # 2. Render Revision Cards (Spaced Repetition Schedule)
                # ---------------------------------------------------------
                if rev_dates:
                    st.markdown("##### Spaced Repetition Schedule")
                    for idx, rev_entry in enumerate(rev_dates):
                        with st.container(border=True):
                            rev_date = rev_entry.get("date")
                            rev_status = rev_entry.get("status", "pending")

                            # INDEX MAPPING: Offset by +1 because Post-Test occupies retention[0]
                            ret_idx = idx + 1
                            attempt = (
                                ret_scores[ret_idx] if ret_idx < len(ret_scores) else {}
                            )

                            score_val = attempt.get("score")
                            quiz_path = attempt.get("quiz_path")

                            # Determine card state
                            if rev_status == "completed" and score_val is not None:
                                score_str = f"{score_val}/10.0"
                                badge = "✅ Completed"
                            elif quiz_path and score_val is None:
                                score_str = "Pending"
                                badge = "🔄 In Progress"
                            else:
                                score_str = "Locked"
                                badge = "⏳ Scheduled"

                            cc1, cc2 = st.columns([3, 1])
                            with cc1:
                                st.markdown(
                                    f"**Phase:** Revision {idx + 1} &nbsp; `{badge}`"
                                )
                                st.write(
                                    f"**Score:** {score_str} &nbsp;&nbsp;|&nbsp;&nbsp; **Date:** {rev_date}"
                                )
                            with cc2:
                                if quiz_path:
                                    st.write("")
                                    if score_val is None:
                                        if st.button(
                                            "▶ Take Quiz",
                                            key=f"t_rev_{topic}_{idx}",
                                            type="primary",
                                            use_container_width=True,
                                        ):
                                            with open(
                                                Path(quiz_path), "r", encoding="utf-8"
                                            ) as f:
                                                quiz_data = json.load(f)

                                            st.session_state.active_quiz_data = {
                                                "phase": "revision",
                                                "quiz": quiz_data,
                                                "saved_path": quiz_path,
                                            }
                                            # FIX: Carry over the exact date for evaluation
                                            st.session_state.simulated_date_str = (
                                                attempt.get("date")
                                            )
                                            st.session_state.quiz_view = "take"
                                            st.rerun()
                                    else:
                                        if st.button(
                                            "View Results",
                                            key=f"v_rev_{topic}_{idx}",
                                            use_container_width=True,
                                        ):
                                            st.session_state.view_quiz_path = quiz_path
                                            st.session_state.quiz_view = "results"
                                            st.rerun()

        with action_col:
            st.markdown("#### Actions")
            with st.container(border=True):
                # Simulator Check: Disable if no revision dates exist
                rev_dates = profile.get("revision_dates", {}).get(topic, [])
                has_revisions = len(rev_dates) > 0

                simulated_date = st.date_input(
                    "Simulate Date",
                    disabled=not has_revisions,
                    help="Active only when spaced repetition revision dates have been generated.",
                )

                st.write("")

                if st.button("▶ Start Quiz", type="primary", use_container_width=True):
                    # 1. Ensure requirements: Toggle notes_reviewed to True automatically
                    profile["topics"][topic]["notes_reviewed"] = True
                    save_profile(st.session_state.student_number, profile)

                    # 2. Check for backend dependencies
                    if "llm" not in st.session_state or "store" not in st.session_state:
                        st.error("Backend not initialized. Please log out and back in.")
                        st.stop()

                    with st.spinner("Denseless is generating your quiz..."):
                        sim_date_str = (
                            simulated_date.strftime("%Y-%m-%d")
                            if simulated_date
                            else None
                        )

                        # 3. Call the router
                        result = handle_quiz_request(
                            student_id=st.session_state.student_id,
                            course="Default",
                            topic=topic,
                            store=st.session_state.store,
                            llm=st.session_state.llm,
                            simulated_today=sim_date_str,
                        )

                        # 4. Handle Router Responses
                        status = result.get("status")

                        if status == "ok":
                            # 1. Determine category based on phase
                            phase = result.get("phase")
                            score_category = (
                                "retention" if phase == "revision" else "comprehension"
                            )

                            # 2. Ensure nested dictionaries and arrays exist
                            if score_category not in profile["scores"]:
                                profile["scores"][score_category] = {}
                            if topic not in profile["scores"][score_category]:
                                profile["scores"][score_category][topic] = []

                            # 3. Calculate attempt number
                            attempt_num = len(profile["scores"][score_category][topic])

                            # 4. Enforce Relative Pathing for Quizzes

                            raw_path = result.get("saved_path")
                            quiz_filename = (
                                Path(raw_path).name if raw_path else "unknown.json"
                            )
                            clean_relative_path = f"data/quizzes/{quiz_filename}"

                            # 5. Append the pending attempt record with the clean path
                            current_date_str = (
                                sim_date_str
                                if sim_date_str
                                else datetime.now().strftime("%Y-%m-%d")
                            )
                            profile["scores"][score_category][topic].append(
                                {
                                    "attempt": attempt_num,
                                    "score": None,  # Will be updated after run_eval_chain
                                    "date": current_date_str,
                                    "quiz_path": clean_relative_path,
                                }
                            )

                            save_profile(st.session_state.student_number, profile)

                            # 6. Move to the take quiz view
                            st.session_state.active_quiz_data = result
                            st.session_state.simulated_date_str = sim_date_str
                            st.session_state.quiz_view = "take"
                            st.rerun()
                        elif status == "no_quiz_due":
                            st.info(
                                f"{result.get('message')} Next date: {result.get('next_quiz_date')}"
                            )
                        elif status == "series_complete":
                            st.success(result.get("message"))
                        elif status == "blocked":
                            st.warning(result.get("message"))
                        else:
                            st.error(
                                result.get("message", "An unexpected error occurred.")
                            )

    # --- VIEW 3: Taking the Quiz ---
    elif st.session_state.quiz_view == "take":
        quiz_payload = st.session_state.active_quiz_data
        topic = st.session_state.quiz_topic
        phase = quiz_payload.get("phase", "Unknown")
        questions = quiz_payload.get("quiz", {}).get("questions", [])
        total_q = len(questions)

        # Initialize pagination index
        if "current_q_idx" not in st.session_state:
            st.session_state.current_q_idx = 0

        current_idx = st.session_state.current_q_idx

        # --- Header ---
        c_head, c_exit = st.columns([4, 1])
        with c_head:
            st.markdown(f"### {topic} - {phase.replace('_', ' ').title()}")
            st.caption(f"Question {current_idx + 1} of {total_q}")
        with c_exit:
            if st.button("❌ Exit Quiz", use_container_width=True):
                st.session_state.active_quiz_data = None
                st.session_state.current_q_idx = 0
                st.session_state.quiz_view = "detail"
                st.rerun()

        # Progress bar
        st.progress((current_idx + 1) / total_q)
        st.write("")

        # --- Question Card Layout ---
        q_data = questions[current_idx]
        q_text = q_data.get("question", "")

        st.markdown(
            f"""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 8px; display: flex; align-items: flex-start; gap: 15px; margin-bottom: 20px;">
            <div style="background-color: #6C5CE7; color: white; border-radius: 8px; width: 45px; height: 45px; display: flex; justify-content: center; align-items: center; font-weight: bold; font-size: 18px; flex-shrink: 0;">
                {current_idx + 1}
            </div>
            <div style="font-size: 18px; font-weight: 600; color: #2D3436; margin-top: 8px;">
                {q_text}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # --- Answer Input ---
        st.markdown(
            "**Your Answer** &nbsp;&nbsp;&nbsp; <span style='color: gray; font-size: 12px; float: right;'>Markdown Supported</span>",
            unsafe_allow_html=True,
        )

        # Load existing answer if they navigated back
        existing_answer = q_data.get("student_answer", "")

        ans_key = f"ans_input_{current_idx}"
        student_input = st.text_area(
            "Answer Input",
            value=existing_answer,
            height=250,
            key=ans_key,
            label_visibility="collapsed",
            placeholder="Type your answer here... Structure your response clearly.",
        )

        st.divider()

        # --- Footer Navigation ---
        c_prev, c_space, c_next = st.columns([1.5, 4, 1.5])

        with c_prev:
            if current_idx > 0:
                if st.button("← Previous", use_container_width=True):
                    # Save current answer before moving
                    questions[current_idx]["student_answer"] = st.session_state[ans_key]
                    st.session_state.current_q_idx -= 1
                    st.rerun()

        with c_next:
            if current_idx < total_q - 1:
                if st.button("Next →", type="primary", use_container_width=True):
                    # Save current answer before moving
                    questions[current_idx]["student_answer"] = st.session_state[ans_key]
                    st.session_state.current_q_idx += 1
                    st.rerun()
            else:
                if st.button(
                    "Finish and Submit →", type="primary", use_container_width=True
                ):
                    # Save the final answer
                    questions[current_idx]["student_answer"] = st.session_state[ans_key]

                    with st.spinner("Evaluating your answers..."):
                        # 1. Resolve exact file path

                        raw_path = quiz_payload.get("saved_path")
                        quiz_filename = Path(raw_path).name
                        quiz_path_obj = Path("data") / "quizzes" / quiz_filename

                        # 2. Write the fully populated quiz dict back to the JSON file
                        with open(quiz_path_obj, "w", encoding="utf-8") as f:
                            json.dump(quiz_payload["quiz"], f, indent=2)

                        # 3. Call the evaluation chain
                        response = run_eval_chain(
                            student_id=st.session_state.student_id,
                            topic=topic,
                            quiz_phase=phase,
                            quiz_path=quiz_path_obj,
                            llm=st.session_state.llm,
                            simulated_date=st.session_state.get("simulated_date_str"),
                        )

                        # 4. Profile Deduplication (Merge the frontend placeholder with the chain's append)
                        # 4. Profile Deduplication (Merge the frontend placeholder with the chain's append)
                        updated_profile = load_profile(st.session_state.student_number)
                        for category in ["comprehension", "retention"]:
                            cat_scores = (
                                updated_profile.get("scores", {})
                                .get(category, {})
                                .get(topic, [])
                            )

                            # Group by quiz_path instead of attempt to catch the duplicate
                            merged_by_path = {}

                            for entry in cat_scores:
                                # Fallback path if missing
                                path_key = entry.get(
                                    "quiz_path", f"data/quizzes/{quiz_filename}"
                                )

                                if path_key not in merged_by_path:
                                    merged_by_path[path_key] = entry.copy()
                                    merged_by_path[path_key]["quiz_path"] = path_key
                                else:
                                    # Merge properties: overwrite nulls with actual values
                                    if entry.get("score") is not None:
                                        merged_by_path[path_key]["score"] = entry[
                                            "score"
                                        ]
                                    if entry.get("date"):
                                        merged_by_path[path_key]["date"] = entry["date"]

                            # Re-build the array and sequentially reassign 'attempt' indices
                            final_scores = list(merged_by_path.values())
                            for i, item in enumerate(final_scores):
                                item["attempt"] = i

                            # Save back to the profile
                            if topic in updated_profile.get("scores", {}).get(
                                category, {}
                            ):
                                updated_profile["scores"][category][topic] = (
                                    final_scores
                                )

                        save_profile(st.session_state.student_number, updated_profile)

                        # 5. Clean up state and return to detail view
                        st.session_state.active_quiz_data = None
                        st.session_state.current_q_idx = 0
                        st.session_state.quiz_view = "detail"

                        st.success("Quiz submitted and graded successfully!")
                        st.rerun()

    # --- VIEW 4: Quiz Results ---
    elif st.session_state.quiz_view == "results":
        quiz_path = st.session_state.get("view_quiz_path")
        topic = st.session_state.quiz_topic

        # Hide standard header to build the custom one

        if not quiz_path or not Path(quiz_path).exists():
            st.error("Quiz file could not be found on disk.")
            if st.button("🔙 Back"):
                st.session_state.quiz_view = "detail"
                st.rerun()
            return

        with open(Path(quiz_path), "r", encoding="utf-8") as f:
            quiz_data = json.load(f)

        questions = quiz_data.get("questions", [])
        total_score = sum(q.get("score", 0) for q in questions)

        # --- Custom Header ---
        h1, h2 = st.columns([4, 1])
        with h1:
            st.markdown(
                f"<h1 style='margin-bottom: 0px;'>{topic}</h1>", unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='color: #636E72; font-size: 14px; margin-top: -10px;'>✅ Quiz Completed &nbsp;•&nbsp; {len(questions)} questions</div>",
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown(
                f"""
            <div style='border: 1px solid #E0E0E0; border-radius: 12px; padding: 15px; text-align: center; font-size: 28px; font-weight: bold;'>
                {total_score}<span style='font-size: 16px; color: #A0A0A0; font-weight: normal;'>/10</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.write("")

        # --- AI Analysis Section ---
        feedback = quiz_data.get("feedback")
        if feedback:
            st.markdown(
                f"""
            <div style='background-color: #F8F5FF; border-left: 4px solid #6C5CE7; padding: 25px; border-radius: 0 8px 8px 0; margin-bottom: 25px;'>
                <div style='color: #6C5CE7; font-size: 11px; font-weight: bold; letter-spacing: 1px; margin-bottom: 15px;'>✨ AI ANALYSIS</div>
                <div style='color: #2D3436; font-size: 15px; line-height: 1.6;'>{feedback}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # --- Weak Topics Section ---
        profile = load_profile(st.session_state.student_number)
        weak_areas = profile.get("topics", {}).get(topic, {}).get("weak_areas", [])

        if weak_areas:
            weak_tags = " ".join(
                [
                    f"<span style='background-color: #FFF; padding: 3px 10px; border-radius: 15px; font-size: 12px; font-weight: 600; border: 1px solid #FFCDCD; margin: 0 4px;'>{w}</span>"
                    for w in weak_areas
                ]
            )
            st.markdown(
                f"""
            <div style='background-color: #FFF5F5; border: 1px solid #FFCDCD; padding: 20px; border-radius: 8px; margin-bottom: 40px; display: flex; align-items: flex-start;'>
                <div style='background-color: #FFCDCD; color: #D63031; width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0; font-size: 18px;'>📉</div>
                <div>
                    <div style='font-weight: bold; font-size: 14px; color: #2D3436; margin-bottom: 5px;'>Weak Topics Updated</div>
                    <div style='font-size: 14px; color: #636E72; line-height: 1.5;'>Based on this quiz, {weak_tags} have been added to your focus areas for future targeted quizzes.</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # --- Question Review ---
        st.markdown("### Question Review")
        st.write("")

        for idx, q in enumerate(questions):
            score = q.get("score", 0)
            student_ans = q.get("student_answer", "")
            model_ans = q.get("model_answer", "")

            with st.container(border=True):
                qc1, qc2 = st.columns([5, 1])
                with qc1:
                    st.markdown(f"**{idx + 1}. {q.get('question')}**")
                with qc2:
                    st.markdown(
                        f"<div style='padding: 4px 8px; border-radius: 4px; font-size: 14px; text-align: right; font-weight: bold;'>Score: {score} / 1.0</div>",
                        unsafe_allow_html=True,
                    )

                st.write("")

                # Student Answer
                st.markdown(
                    f"""
                <div style='border: 1px solid #E9ECEF; padding: 15px; border-radius: 6px; margin-bottom: 15px;'>
                    <div style='color: #FF7675; font-size: 10px; font-weight: bold; margin-bottom: 8px; letter-spacing: 0.5px;'>YOUR ANSWER</div>
                    <div style='font-size: 14px; line-height: 1.5;'>{student_ans}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Correct Answer
                st.markdown(
                    f"""
                <div style='border: 1px solid #81ECEC; padding: 15px; border-radius: 6px; margin-bottom: 10px;'>
                    <div style='color: #00CEC9; font-size: 10px; font-weight: bold; margin-bottom: 8px; letter-spacing: 0.5px;'>CORRECT ANSWER</div>
                    <div style='font-size: 14px; line-height: 1.5;'>✓ {model_ans}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.write("")
        st.write("")

        # --- Footer Actions ---
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Back to Study", use_container_width=True):
                st.session_state.quiz_view = "detail"
                st.session_state.view_quiz_path = None
                st.rerun()
        with b2:
            if st.button("Back to Dashboard", use_container_width=True):
                st.session_state.page = "Dashboard"
                st.rerun()


@st.dialog("Topic Details")
def render_topic_modal(topic: str, profile: dict):
    st.markdown("##### Strong and Weak Areas")

    topic_data = profile.get("topics", {}).get(topic, {})
    weak_areas = topic_data.get("weak_areas", [])
    strong_areas = topic_data.get("strong_areas", [])

    st.markdown("**Weak Areas**")
    if weak_areas:
        weak_tags = " ".join(
            [
                f"<span style='background-color: #FFCDCD; color: #D63031; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 500; margin-right: 6px; display: inline-block; margin-bottom: 6px;'>{w}</span>"
                for w in weak_areas
            ]
        )
        st.markdown(weak_tags, unsafe_allow_html=True)
    else:
        st.caption("No weak areas identified yet.")

    st.write("")
    st.markdown("**Strong Areas**")
    if strong_areas:
        strong_tags = " ".join(
            [
                f"<span style='background-color: #6C5CE7; color: white; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 500; margin-right: 6px; display: inline-block; margin-bottom: 6px;'>{s}</span>"
                for s in strong_areas
            ]
        )
        st.markdown(strong_tags, unsafe_allow_html=True)
    else:
        st.caption("No strong areas identified yet.")

    st.divider()

    st.markdown("##### Comprehension & Retention Scores")

    # Aggregate scores into a clean list for the table
    table_data = []

    comp_scores = profile.get("scores", {}).get("comprehension", {}).get(topic, [])
    ret_scores = profile.get("scores", {}).get("retention", {}).get(topic, [])

    for c in comp_scores:
        if c.get("score") is not None:
            c_type = "Pre" if c.get("attempt") == 0 else "Post"
            table_data.append(
                {
                    "Test": "Comprehension",
                    "Type": c_type,
                    "Score": f"{c.get('score')}/10",
                }
            )

    for r in ret_scores:
        if r.get("score") is not None:
            attempt_num = r.get("attempt")
            if attempt_num > 0:  # Skip the inherited post-test at index 0
                table_data.append(
                    {
                        "Test": "Retention",
                        "Type": str(attempt_num),
                        "Score": f"{r.get('score')}/10",
                    }
                )

    if table_data:
        st.table(table_data)
    else:
        st.info("No completed quizzes found for this topic.")

    if st.button("Dismiss", type="primary", use_container_width=True):
        st.rerun()


@st.dialog("Token History")
def render_token_modal(profile: dict):

    st.markdown("Select a date range (maximum 14 days) to view your token consumption.")

    token_history = profile.get("token_history", [])

    # Aggregate token data by date
    agg_data = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})
    for entry in token_history:
        date_str = entry.get("timestamp", "")[:10]
        if date_str:
            agg_data[date_str]["input"] += entry.get("input_tokens", 0)
            agg_data[date_str]["output"] += entry.get("output_tokens", 0)
            agg_data[date_str]["total"] += entry.get("total_tokens", 0)

    today = datetime.now().date()
    default_start = today - timedelta(days=7)

    # Date range selector
    date_range = st.date_input(
        "Date Range", value=(default_start, today), max_value=today
    )

    st.write("")

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range

        # Enforce 14-day limit
        delta = end_date - start_date
        if delta.days > 14:
            st.warning("Please select a date range of 14 days or less.")
        else:
            table_data = []
            for d_str, counts in agg_data.items():
                d_obj = datetime.strptime(d_str, "%Y-%m-%d").date()
                if start_date <= d_obj <= end_date:
                    table_data.append(
                        {
                            "Date": d_str,
                            "Input Tokens": f"{counts['input']:,}",
                            "Output Tokens": f"{counts['output']:,}",
                            "Total Tokens": f"{counts['total']:,}",
                        }
                    )

            if table_data:
                table_data.sort(key=lambda x: x["Date"], reverse=True)
                st.table(table_data)
            else:
                st.info("No token usage found in this date range.")
    else:
        st.info("Please select a start and end date.")

    if st.button("Dismiss", use_container_width=True):
        st.rerun()


def render_analytics():
    st.markdown("# Learner Profile & Analytics")
    st.markdown(
        "<div style='color: #636E72; margin-top: -15px; margin-bottom: 25px;'>Track your academic progress and manage your learning preferences.</div>",
        unsafe_allow_html=True,
    )

    profile = load_profile(st.session_state.student_number)
    if not profile:
        st.error("Profile not found.")
        return

    top_col1, top_col2 = st.columns([2, 1], gap="large")

    # --- TOP LEFT: Topic Performance ---
    with top_col1:
        st.markdown("### Topic Performance")
        with st.container(border=True):
            topics = profile.get("topics", {})
            if not topics:
                st.info("No topics ingested yet.")
            else:
                for t_name in topics.keys():
                    if st.button(
                        t_name, key=f"topic_btn_{t_name}", use_container_width=True
                    ):
                        render_topic_modal(t_name, profile)

    # --- TOP RIGHT: Learning Pace ---
    with top_col2:
        st.markdown("### ⚙️ Learning Pace")
        with st.container(border=True):
            st.markdown(
                "<div style='font-size: 14px; color: #636E72; margin-bottom: 15px;'>Adjust how AI condenses notes based on your preferred review speed.</div>",
                unsafe_allow_html=True,
            )

            current_pace = profile.get("learning_pace", "average")

            pace_options = {
                "fast": "Fast (Summary) - High-level bullet points",
                "average": "Average (Standard) - Balanced mix of key concepts",
                "slow": "Slow (Deep Dive) - Detailed step-by-step breakdowns",
            }

            # Map current value to index for the radio button
            pace_keys = list(pace_options.keys())
            default_index = (
                pace_keys.index(current_pace) if current_pace in pace_keys else 1
            )

            selected_pace_label = st.radio(
                "Pace Selection",
                options=list(pace_options.values()),
                index=default_index,
                label_visibility="collapsed",
            )

            # Map back to the underlying key
            selected_pace = pace_keys[
                list(pace_options.values()).index(selected_pace_label)
            ]

            # Save if changed
            if selected_pace != current_pace:
                profile["learning_pace"] = selected_pace
                save_profile(st.session_state.student_number, profile)
                st.toast("Learning pace updated!")

    st.write("")

    # --- MIDDLE: Token History ---
    c_title, c_btn = st.columns([4, 1])
    with c_title:
        st.markdown("### Token History")
    with c_btn:
        if st.button("View All", key="view_all_tokens", use_container_width=True):
            render_token_modal(profile)

    with st.container(border=True):
        token_history = profile.get("token_history", [])
        if not token_history:
            st.info(
                "Token tracking metrics will populate here as you interact with the AI."
            )
        else:
            from collections import defaultdict

            agg_data = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})

            for entry in token_history:
                date_str = entry.get("timestamp", "")[:10]
                if date_str:
                    agg_data[date_str]["input"] += entry.get("input_tokens", 0)
                    agg_data[date_str]["output"] += entry.get("output_tokens", 0)
                    agg_data[date_str]["total"] += entry.get("total_tokens", 0)

            sorted_dates = sorted(agg_data.keys(), reverse=True)

            # Extract only the 4 most recent dates
            table_data = []
            for d_str in sorted_dates[:4]:
                counts = agg_data[d_str]
                table_data.append(
                    {
                        "Date": d_str,
                        "Input Tokens": f"{counts['input']:,}",
                        "Output Tokens": f"{counts['output']:,}",
                        "Total Tokens": f"{counts['total']:,}",
                    }
                )

            if table_data:
                st.table(table_data)

    st.write("")

    # --- BOTTOM: Revision Feedback ---
    st.markdown("### Revision Feedback")

    # Gather only the single most recent completed feedback entry per topic
    latest_feedback_per_topic = []
    rev_dates = profile.get("revision_dates", {})

    for t_name, entries in rev_dates.items():
        topic_feedbacks = []
        for entry in entries:
            if entry.get("status") == "completed" and entry.get("feedback"):
                topic_feedbacks.append(
                    {
                        "topic": t_name,
                        "date": entry.get("date"),
                        "feedback": entry.get("feedback"),
                    }
                )

        if topic_feedbacks:
            # Sort this specific topic's feedback by date (newest first)
            topic_feedbacks.sort(key=lambda x: x["date"], reverse=True)
            # Append only the first one
            latest_feedback_per_topic.append(topic_feedbacks[0])

    if not latest_feedback_per_topic:
        with st.container(border=True):
            st.info(
                "No revision feedback available yet. Complete a revision quiz to generate AI feedback."
            )
    else:
        # Sort the final curated list by date descending so the newest overall is at the top
        latest_feedback_per_topic.sort(key=lambda x: x["date"], reverse=True)

        for item in latest_feedback_per_topic:
            with st.container(border=True):
                st.markdown(
                    f"<div style='color: #2D3436; font-size: 15px; margin-bottom: 10px;'>\"{item['feedback']}\"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='color: #8395A7; font-size: 12px; font-weight: 500;'>{item['topic']} • {item['date']}</div>",
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------
# Post-login shell (sidebar nav placeholder — built out page by page)
# --------------------------------------------------------------------------
def render_app_shell():
    st.set_page_config(page_title="Denseless", page_icon="📘", layout="wide")

    with st.sidebar:
        st.markdown(f"**Logged in as:** `{st.session_state.student_id}`")
        st.divider()
        page = st.radio(
            "Navigate",
            ["Dashboard", "Library", "AI Workspace", "Quiz", "Analytics"],
            label_visibility="collapsed",
        )
        st.divider()
        if st.button("Log out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.student_number = None
            st.session_state.student_id = None
            st.rerun()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Library":
        render_library()
    elif page == "AI Workspace":
        render_workspace()
    elif page == "Quiz":
        render_quiz()
    elif page == "Analytics":
        render_analytics()


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
if st.session_state.authenticated:
    render_app_shell()
else:
    render_login()
