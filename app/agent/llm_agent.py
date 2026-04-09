import sys
from pathlib import Path

# The below code is used to access packages from the root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agent.rag.ingestion.data_ingestion import process_and_load_file

docs = process_and_load_file("app/pdfs/Interconnection Structures.pdf")
print(docs)
