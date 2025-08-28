import os
import shutil
from pathlib import Path

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from sentence_transformers import CrossEncoder
import torch

BASE_DIR   = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "vector_db"
DATA_DIR   = BASE_DIR / "data"

print(BASE_DIR)
print(INDEX_PATH)
print(DATA_DIR)

def vectorize_pdf():
    logging.warning("vectorize_pdf from %s  DATA_DIR=%s", __file__, DATA_DIR)
    load_dotenv()
    #preprocessed_directory = #os.path.join("data/")
    preprocessed_directory = DATA_DIR
    files = [os.path.join(preprocessed_directory, f) for f in os.listdir(preprocessed_directory) if f.endswith(".pdf")]
    if not files:
        logging.info("No PDF files found in data directory. Skipping vectorization.")
        return

    docs = []

    for file in files:
        with open(file, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

            # Create Document with metadata (optional: include filename, etc.)
            doc = Document(page_content=text, metadata={"source": os.path.basename(file)})
            docs.append(doc)

    #INDEX_PATH = Path("vector_db")

    # Embeddings
    #embeddings = HuggingFaceEmbeddings(model=EMBED_MODEL, show_progress=True)
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2", show_progress=True)
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key="AIzaSyCqsDNnpIT1fXj-ksaFf90_0A1BSL8hu94")

    vector_store = FAISS.from_documents(docs, embeddings, normalize_L2=True)

    # ---- move underlying index to GPU ----
    # res = FAISS.StandardGpuResources()
    # gpu_index = FAISS.index_cpu_to_gpu(res, 0, vector_store.index)
    # vector_store.index = gpu_index        # swap in the GPU handle
    #
    # vector_store.index = FAISS.index_gpu_to_cpu(vector_store.index)
    vector_store.save_local(INDEX_PATH)
    print("end")

def search_similarity(query):
    if not (INDEX_PATH / "index.faiss").exists():
        vectorize_pdf()
    #INDEX_PATH = "vector_db"

    # Recreate the embeddings object
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2", show_progress=True)
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key="AIzaSyCqsDNnpIT1fXj-ksaFf90_0A1BSL8hu94")

    # Load the persisted vector store
    if not (INDEX_PATH / "index.faiss").exists():
        raise FileNotFoundError("Missing FAISS index at vector_db/index.faiss. Run vectorize_pdf() first.")
    vector_store = FAISS.load_local(INDEX_PATH, embeddings, normalize_L2=True, allow_dangerous_deserialization=True)

    # # move to GPU once, cache the handle
    # res = FAISS.StandardGpuResources()
    # vector_store.index = FAISS.index_cpu_to_gpu(res, 0, vector_store.index)

    # Now do a similarity search!
    query_vector = embeddings.embed_query(query)
    raw_results = vector_store.similarity_search_with_score_by_vector(query_vector, k=100)

    short_list = raw_results[:8]

    # -- cross-encoder rerank --------------------------------------------------
    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", activation_fn=torch.nn.Sigmoid())  # tiny & fast
    ce_inputs = [(query, doc.page_content[:50000])  # clip long docs
                 for doc, _ in short_list]
    ce_scores = ce_model.predict(ce_inputs)  # higher = more relevant

    # Combine and sort by CE score (desc). Fall back on dense distance tie-break.
    combined = [
        (doc, dist, float(ce_score))  # ensure JSON serialisable
        for (doc, dist), ce_score in zip(short_list, ce_scores)
    ]
    combined.sort(key=lambda x: (x[2], -x[1]), reverse=True)

    ce_results = [
        {
            "source": doc.metadata.get("source", "N/A"),
            "score": float(round(dist, 4)),  # FAISS L2 distance (lower is better)
            "cross_score": round(ce_score, 4),  # Cross-encoder relevance (higher is better)
            "content": doc.page_content[:40000].strip()
        }
        for doc, dist, ce_score in combined
        if dist < 1.0  # keep only "good" matches (optional)
    ]
    return ce_results
