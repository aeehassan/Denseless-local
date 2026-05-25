import sys
import pandas as pd
import os
from pathlib import Path

# The below code is used to access packages from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

from qpp.agent.rag.ingestion.vector_store import get_vector_store, add_documents_to_chroma
from qpp.agent.rag.ingestion.data_ingestion import process_and_load_file
from qpp.agent.rag.ingestion.embeddings import get_embedding_model, generate_embeddings
from langchain_core.documents import Document
from qpp.agent.rag.retrieval.retriever import get_semantic_chunks, get_topic_chunks
from typing import List
from qpp.agent.rag.chains.qa_chain import run_qa_chain, list_conversations, view_ltm
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json


def format_chunks(chunks: List[Document]) -> str:
    if not chunks:
        return ""
    return "\n".join(f"{i + 1}. {doc.page_content}" for i, doc in enumerate(chunks))


load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

embedder = get_embedding_model()
store = get_vector_store(student_id="1019", embedder=embedder)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    project="denseless",
    location="us-central1",
    vertexai=True,
)

questions = [
    "What specific elements are defined by the Instruction Set Architecture (ISA)?",
    "Why might computer manufacturers offer multiple models with the same architecture but different organizations?",
    "Why is the hierarchical nature of computers essential, and why is the top-down approach effective in describing it?",
    "In the context of data movement, what is the difference between input/output (I/O) processes and data communications?",
    "What are the four main structural components of a traditional single-processor computer?",
    "What are the major internal structural components of the Central Processing Unit (CPU)?",
    'How is a "core" defined within the context of a multicore computer?',
    'What is the physical distinction between a "processor" and a "core"?',
    "What is the fundamental function of a printed circuit board (PCB) or motherboard in a computer system?",
    "What are the three primary functional elements found inside a single core?",
    "Describe the L1, L2, and L3 caches?",
    "What are the three primary functional elements found inside a single core?",
    "What are the three characteristics of a Von Neumann architecture computer system?",
    'What is the "Von Neumann bottleneck" and why does it fundamentally limit CPU efficiency?',
    "How does a Von Neumann computer distinguish between data and instructions if they reside in the same memory?",
]

json_eval_dataset = {}

for i, q in enumerate(questions, start=1):
    key = str(i).zfill(3)

    chunks = get_semantic_chunks(
        query=q,
        store=store,
        score_threshold=0.4,
        top_k=5,
        topic="COMPUTER ARCHITECTURE AND ORGANIZATION I",
        course="CSC 315",
        condensed=False,
    )
    formatted_chunks = format_chunks(chunks)

    response = run_qa_chain(
        "student_1019",
        q,
        "COA_chat_1",
        False,
        store,
        llm,
        "fast",
        "CSC 315Week OneTTTT2.pdf",
        "CSC 315",
    )
    qa_output = response.content["answer"]

    json_eval_dataset[key] = {
        "student_question": q,
        "source_chunks": formatted_chunks,
        "qa_output": qa_output,
    }

output_path = Path(__file__).parent / "eval_dataset.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_eval_dataset, f, indent=4, ensure_ascii=False)

print(f"Successfully completed. Exported to {output_path}")
