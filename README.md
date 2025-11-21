<div align="center">
  
# ⚛️ Quantum_Education_Policy  
### *AI + Quantum Computing for Smarter Education Policy Recommendations*

---

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-brightgreen?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PennyLane](https://img.shields.io/badge/Quantum-PennyLane-purple?logo=pennylane)](https://pennylane.ai)
[![scikit-learn](https://img.shields.io/badge/NLP-scikit--learn-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)]()

---

</div>

## 🧠 Overview

**Quantum_Education_Policy** (also called *PolicyNav*) is a **FastAPI-powered NLP application** enhanced with **Quantum Computing** via *PennyLane*.  
It recommends the most relevant **education policies** for a given query using **TF-IDF + Quantum Similarity Kernels**.

This project explores how **Quantum Machine Learning (QML)** can improve text understanding and information retrieval for **policy analysis** and **decision-making** in the education sector.

---

## 🚀 Features

✅ **AI-driven policy recommendations** using classical + quantum NLP  
⚛️ **Quantum kernel similarity** via PennyLane quantum circuits  
🌐 **Interactive web interface** built with FastAPI + HTML templates  
📊 **Dynamic data visualization** using Chart.js  
🧰 **TF-IDF-based vectorization** with Scikit-learn  
💾 **Pre-trained models stored with Joblib**  
🎯 **Modular, lightweight, and extensible design**

---

## 🧩 Tech Stack

| Layer | Technologies |
|:------|:--------------|
| **Backend** | FastAPI, Uvicorn |
| **Quantum Computing** | PennyLane, PennyLane-Numpy |
| **Machine Learning / NLP** | Scikit-learn (TF-IDF, Cosine Similarity) |
| **Data Handling** | Pandas, NumPy |
| **Frontend** | HTML5, CSS3, Jinja2, Chart.js |
| **Serialization** | Joblib |
| **Runtime** | Python 3.10 |

---

## 🗂️ Project Structure

```

Quantum_Education_Policy/
├── app.py                        # Main FastAPI + Quantum NLP backend
├── templates/
│   └── index.html                # Frontend with interactive visualization
├── static/                       # Optional: CSS / JS / image assets
├── policy_vectorizer.pkl         # Trained TF-IDF vectorizer
├── policy_tfidf_matrix.pkl       # TF-IDF matrix and policy dataset
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
└── .gitattributes                # Git configuration

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/Quantum_Education_Policy.git
cd Quantum_Education_Policy
````

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 📦 requirements.txt

```
fastapi==0.110.0
uvicorn[standard]==0.27.0
pandas==1.5.3
numpy==1.24.4
scikit-learn==1.2.2
joblib==1.2.0
pennylane==0.33.0
pennylane-numpy==0.33.0
jinja2==3.1.2
textwrap3==0.9.2
chartjs==3.9.1
```

---

## ▶️ Running the App

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open your browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

You’ll see a sleek web UI where you can enter education-related queries like:

> "Improve digital education in rural schools"
> and receive the top matching policy recommendations.

---

## 🧮 Core Architecture

### 🔹 NLP + Quantum Workflow

1. **TF-IDF Vectorization:** Text is transformed into weighted feature vectors.
2. **Quantum Embedding:** Encodes TF-IDF vectors as quantum states via amplitude embedding.
3. **Quantum Kernel:** Measures state similarity using quantum circuits.
4. **Ranking:** Policies are sorted by quantum similarity scores.

### 🔹 Backend (app.py)

* Loads pre-trained TF-IDF model + matrix from `.pkl` files
* Processes user input and computes similarity
* Renders top results dynamically using Jinja2 templates

### 🔹 Frontend (index.html)

* Clean interface with a search bar
* Displays policy cards with summaries
* Shows **region-wise** and **status-wise** charts

---

## 📊 Example Output

**User Query:**

> “Enhance rural higher education accessibility”

| Rank | Policy Title                        | Region | Year | Score |
| ---- | ----------------------------------- | ------ | ---- | ----- |
| 1    | Rural Education Development Policy  | South  | 2021 | 0.89  |
| 2    | Higher Learning Digital Access Plan | East   | 2020 | 0.84  |
| 3    | Inclusive Education Strategy        | West   | 2022 | 0.81  |

---
## Quantum_Education_Policy NLP Model

### Dashboard

<img width="1863" height="1023" alt="Screenshot 2025-10-31 202354" src="https://github.com/user-attachments/assets/1e4c0d52-7787-4732-9f0e-a8d8bcca236a" />

### NLP Interface

<img width="1798" height="1042" alt="Screenshot 2025-10-30 191032" src="https://github.com/user-attachments/assets/eafefed8-6cb8-4b5d-a13c-3979519d7508" />


## 🎨 UI Preview

> FastAPI-powered web interface for interactive AI-driven policy search.

* 🔍 Query-based recommendation search
* 🧾 Policy cards with summaries and metadata
* 📈 Dynamic pie and bar charts via Chart.js
* 💡 Responsive design for desktop and mobile

*(You can add a screenshot here later, e.g., `/static/screenshot.png`)*

---

## 🧭 Command Reference

| Task        | Command                           |
| ----------- | --------------------------------- |
| Create venv | `python -m venv venv`             |
| Activate    | `venv\Scripts\activate`           |
| Install     | `pip install -r requirements.txt` |
| Run App     | `uvicorn app:app --reload`        |
| Access      | `http://127.0.0.1:8000`           |

---

## 💡 Future Enhancements

* 🔬 Integrate **Quantum Kernel Learning (QKL)** for dynamic similarity tuning
* 🌍 Support **multilingual policy queries**
* 💾 Add **database storage** (PostgreSQL / MongoDB)
* 🧠 Enhance **semantic similarity** using transformer embeddings
* ☁️ Deploy on **Render**, **Hugging Face Spaces**, or **AWS Lambda**

---

## 👨‍💻 Author

**Pradeesh Vasu**          
📧 *E-mail id:* [pradeeshvasu22@gmail.com]                              
🌐 *github:* [https://github.com/PradeeshVasu]                              
💼 *LinkedIn:* [linkedin.com/in/pradeesh-vasu-03486b319](https://www.linkedin.com/in/pradeesh-vasu-03486b319)                            
**Sai Vadalakonda**                     
📧 *E-mail id:* [sai2592004@gmail.com]            
🌐 *github:* [https://github.com/sai-vadlakonda]                  
💼 *LinkedIn:* [https://www.linkedin.com/in/sai-vadlakonda/]                              
**Sai Surya**                        
**Ekta Sharma**  
📧 *E-mail id:* [ektasharma4462@gmail.com]                              
🌐 *github:* [https://github.com/ekta-240]                             
💼 *LinkedIn:*     [https://www.linkedin.com/in/ekta--sharma24/]                 
**Pavan Kumar**                                      


 


---

## 📜 License

Licensed under the **MIT License** — you are free to use, modify, and distribute with attribution.

---

<div align="center">

> *“Bridging Classical AI and Quantum Computing to shape better education policy intelligence.”*

