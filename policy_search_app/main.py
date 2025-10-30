from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# The heavy ML / quantum libraries are imported lazily so the app can start
try:
    from pydantic import BaseModel
except Exception:
    BaseModel = object

try:
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit.circuit import QuantumCircuit, ParameterVector
except Exception:
    joblib = None
    pd = None
    np = None
    TfidfVectorizer = None
    FidelityQuantumKernel = None
    QuantumCircuit = None
    ParameterVector = None

# ----- Initialize FastAPI -----
app = FastAPI(title="Quantum Policy Search API")

# ----- Paths -----
MODELPATH = "artifacts/quantum_policy_kernel.pkl"
MATRIXPATH = "artifacts/quantum_policy_matrix.pkl"

# ----- Load Quantum Kernel & Dataset -----
quantum_kernel = None
full_df = None

if joblib is not None:
    try:
        quantum_kernel = joblib.load(MODELPATH)
    except Exception:
        quantum_kernel = None
    try:
        saveddata = joblib.load(MATRIXPATH)
        full_df = saveddata.get('df') if isinstance(saveddata, dict) else None
    except Exception:
        full_df = None

# ----- Preprocessing Function -----
def preprocess_text(title, fulltext, stakeholders):
    return f"{str(title)}. {str(fulltext)}. Stakeholders: {str(stakeholders).lower()}"

# ----- Ensure columns exist -----
if full_df is not None and pd is not None:
    required_columns = ['title', 'fulltext', 'stakeholders']
    for col in required_columns:
        if col not in full_df.columns:
            full_df[col] = ""

    full_df['textfornlp'] = full_df.apply(
        lambda x: preprocess_text(x.get('title', ''), x.get('fulltext', ''), x.get('stakeholders', '')),
        axis=1
    )

    if TfidfVectorizer is not None:
        vectorizer = TfidfVectorizer(max_features=8)
        try:
            vectorizer.fit(full_df['textfornlp'])
        except Exception:
            vectorizer = None
    else:
        vectorizer = None
else:
    vectorizer = None

# ----- Request Model -----
class PolicyQuery(BaseModel):
    query: str
    top_n: int = 3

# ----- Helper to convert numpy types -----
def convert_numpy(obj):
    import numpy as np
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    else:
        return obj

# ----- Endpoint -----
@app.post("/quantum_search")
def quantum_policy_search(request: PolicyQuery):
    query_text = preprocess_text(request.query, request.query, "All")

    if vectorizer is None or quantum_kernel is None or full_df is None or np is None:
        return JSONResponse(status_code=503, content={
            "error": "Model or data not available. Ensure dependencies are installed and artifacts exist.",
        })

    X_query_tfidf = vectorizer.transform([query_text]).toarray()
    X_query_norm = np.pi * X_query_tfidf / np.max(X_query_tfidf) if np.max(X_query_tfidf) != 0 else X_query_tfidf
    X_full_tfidf = vectorizer.transform(full_df['textfornlp']).toarray()
    X_full_norm = np.pi * X_full_tfidf / np.max(X_full_tfidf) if np.max(X_full_tfidf) != 0 else X_full_tfidf

    sim_scores = quantum_kernel.evaluate(X_query_norm, X_full_norm)[0]
    top_indices = np.argsort(sim_scores)[::-1][:request.top_n]

    results = []
    for idx in top_indices:
        row = full_df.iloc[idx]
        results.append({
            "policy_id": str(row.get("policyid", "")),
            "title": str(row.get("title", "")),
            "sector": str(row.get("sector", "")),
            "region": str(row.get("region", "")),
            "year": str(row.get("year", "")),
            "impactscore": str(row.get("impactscore", "")),
            "similarity_score": round(float(sim_scores[idx]), 4),
            "summary": str(row.get("summary", "")),
            "goals": str(row.get("goals", ""))
        })

    # ✅ Convert any NumPy values to native Python types
    safe_results = convert_numpy(results)
    return {"top_policies": safe_results}

# ----- Static & Templates -----
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    """Serve the search page (templates/index.html)."""
    return templates.TemplateResponse("index.html", {"request": request})
