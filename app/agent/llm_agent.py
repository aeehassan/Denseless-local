import sys
from pathlib import Path

# The below code is used to access packages from the root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agent.rag.ingestion.data_ingestion import process_and_load_file
from app.agent.rag.ingestion.embeddings import get_embedding_model

# 1. Test Ingestion
# docs = process_and_load_file("app/pdfs/Interconnection Structures.pdf")
# print(docs)

# 2. Test Embedding
# embedder = get_embedding_model()
# test_vec = embedder.embed_query("This is a test of the embedding system.")
# print(f"Generated vector of length {len(test_vec)}")
