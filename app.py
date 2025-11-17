from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import textwrap
import warnings

# Suppress RuntimeWarning for cleaner output (optional)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------- FastAPI App Setup ----------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------- Load Model + Data ----------
MODEL_PATH = "policy_vectorizer.pkl"  # Updated file path
MATRIX_PATH = "policy_tfidf_matrix.pkl"  # Updated file path

try:
    vectorizer = joblib.load(MODEL_PATH)
    data = joblib.load(MATRIX_PATH)
    tfidf_matrix = data["tfidf_matrix"]
    df = data["df"]
    normalized_vectors = data["normalized_vectors"]
except Exception as e:
    print(f"Error loading model/data: {e}")
    raise

# Verify required columns in DataFrame
required_columns = ["title", "policy_id", "region", "year", "status", "full_text"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Warning: Missing columns in DataFrame: {missing_columns}")
    for col in missing_columns:
        df[col] = "Unknown"

# Quantum device setup (3 qubits for 8 features)
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum feature map (amplitude encoding)
def feature_map(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        vec = np.ones_like(vec) / np.sqrt(len(vec))
    qml.AmplitudeEmbedding(vec, wires=range(n_qubits), normalize=True)
    qml.BasicEntanglerLayers(np.random.random((2, n_qubits)), wires=range(n_qubits))

# Quantum kernel circuit
@qml.qnode(dev)
def kernel_circuit(v1, v2):
    feature_map(v1)
    qml.adjoint(feature_map)(v2)
    return qml.probs(wires=range(n_qubits))

# Recompute normalized vectors with zero-norm handling
norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1  # Avoid division by zero
normalized_vectors = tfidf_matrix / norms

# Verify and fix zero-norm vectors
for i, vec in enumerate(normalized_vectors):
    if np.all(vec == 0):
        print(f"Warning: Zero-norm vector detected at index {i}. Replacing with uniform vector.")
        normalized_vectors[i] = np.ones_like(vec) / np.sqrt(len(vec))

# ---------- Quantum Search Function ----------
def search_policies_quantum(query: str, top_k: int = 3):
    try:
        query_text = query.lower()
        query_tfidf = vectorizer.transform([query_text]).toarray()[0]
        norm = np.linalg.norm(query_tfidf)
        query_vec = query_tfidf / norm if norm > 0 else np.ones_like(query_tfidf) / np.sqrt(len(query_tfidf))
        
        sims = []
        for i, vec in enumerate(normalized_vectors):
            try:
                sim = kernel_circuit(query_vec, vec)[0]
                sims.append(sim)
            except Exception as e:
                print(f"Error computing kernel for document {i}: {e}. Skipping.")
                sims.append(0.0)
        
        sims = np.array(sims)
        top_idx = sims.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_idx:
            row = df.iloc[idx]
            results.append({
                "title": row["title"],
                "policy_id": row["policy_id"],
                "region": row["region"],
                "year": row["year"],
                "status": row["status"],
                "summary": textwrap.shorten(row["full_text"], width=250, placeholder="..."),
                "score": round(float(sims[idx]), 3)  # Convert tensor to float before rounding
            })
        return results, None
    except Exception as e:
        return [], f"Error processing query: {str(e)}"

# ---------- FastAPI Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None, "error": None})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    results, error = search_policies_quantum(query)
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query, "error": error})