from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_7Lqy9p_ApY292C4iAJ4qGdmcTJG6QVYVjZSDSYALjrv4bTESULEHuUvPR6gmHjUP4DZMSX")
index = pc.Index("final-legal")

import pdfplumber
import uuid
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# ✅ Step 1: Initialize API Keys
PINECONE_API_KEY = "pcsk_7Lqy9p_ApY292C4iAJ4qGdmcTJG6QVYVjZSDSYALjrv4bTESULEHuUvPR6gmHjUP4DZMSX"
INDEX_NAME = "final-legal"

# ✅ Step 2: Initialize Pinecone and Sentence Transformers
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")  # Change env if needed
model = SentenceTransformer("all-MiniLM-L6-v2")  # Uses Hugging Face instead of OpenAI

# ✅ Step 3: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

pdf_text = extract_text_from_pdf("COI.pdf")
print(f"Extracted {len(pdf_text)} characters from PDF.")

# ✅ Step 4: Split Text into Chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(pdf_text, chunk_size=1000)  # Adjust chunk size if needed
print(f"Split into {len(chunks)} chunks.")

# ✅ Step 5: Convert Chunks into Embeddings using Sentence Transformers
embeddings = model.encode(chunks, convert_to_list=True)
print("Generated embeddings.")

# ✅ Step 6: Upload to Pinecone
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=len(embeddings[0]), metric="cosine")

index = pc.Index(INDEX_NAME)
vectors = [(str(uuid.uuid4()), emb, {"text": chunks[i]}) for i, emb in enumerate(embeddings)]
index.upsert(vectors)
print("✅ Uploaded embeddings to Pinecone.")

