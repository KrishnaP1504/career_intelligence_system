# 🧠 Career Intelligence System

> An AI-powered career analysis platform that analyzes resumes, predicts job roles, and provides actionable skill gap insights.

---

## ✨ Features

- 📄 **Resume Analysis** — Upload and parse resumes to extract skills, experience, and domain keywords
- 🔮 **Job Role Prediction** — ML-powered classifier predicts the most suitable job roles with confidence scores
- 📊 **Skill Gap Analysis** — Compares your current skills against target role requirements
- 💡 **Explainable AI** — SHAP/LIME-style explanations for predictions
- 🗂️ **History Tracking** — Save and revisit past analysis sessions via MongoDB
- 🌐 **REST API** — Flask-based backend with clean API endpoints

---

## 🏗️ Project Structure

```
career_intelligence_system/
├── backend/                  # Flask REST API
│   ├── api/
│   │   └── routes.py         # API endpoints
│   ├── core/
│   │   └── database.py       # MongoDB connection
│   ├── models/
│   │   └── schemas.py        # Pydantic/data schemas
│   ├── main.py               # App entry point
│   ├── requirements.txt      # Backend dependencies
│   └── Dockerfile
│
├── frontend/                 # Streamlit UI
│   ├── src/
│   │   ├── predictor.py      # ML prediction logic
│   │   ├── nlp_processor.py  # Resume NLP processing
│   │   ├── skill_analyzer.py # Skill gap analysis
│   │   ├── explainer.py      # Model explainability
│   │   └── domain_matcher.py # Domain classification
│   ├── app.py                # Streamlit main app
│   ├── api_client.py         # Backend API client
│   ├── requirements.txt      # Frontend dependencies
│   └── Dockerfile
│
└── scripts/                  # Training & data scripts
    ├── train_models.ipynb    # Model training notebook
    └── data/                 # Training data (not tracked)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- MongoDB (local or Atlas)
- Docker (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/career_intelligence_system.git
cd career_intelligence_system
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your MongoDB URI

python main.py
```

Backend runs at: `http://localhost:5000`

### 3. Frontend Setup

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Frontend runs at: `http://localhost:8501`

### 4. Run with Docker

```bash
# Backend
cd backend
docker build -t career-intel-backend .
docker run -p 5000:5000 career-intel-backend

# Frontend
cd frontend
docker build -t career-intel-frontend .
docker run -p 8501:8501 career-intel-frontend
```

---

## 🧪 Training Models

The trained models (`*.pkl`) are not tracked in git due to file size. To retrain:

1. Add your resume dataset to `scripts/data/raw/`
2. Open and run `scripts/train_models.ipynb`
3. Trained models will be saved to `frontend/saved_models/`

---

## 🛠️ Tech Stack

| Layer     | Technology                          |
|-----------|-------------------------------------|
| Frontend  | Streamlit, Python                   |
| Backend   | Flask, Python                       |
| Database  | MongoDB                             |
| ML/NLP    | scikit-learn, spaCy, NLTK           |
| Models    | Random Forest Classifier            |
| DevOps    | Docker                              |

---

## 📡 API Endpoints

| Method | Endpoint              | Description                  |
|--------|-----------------------|------------------------------|
| POST   | `/api/analyze`        | Analyze resume & predict role|
| GET    | `/api/history`        | Fetch analysis history       |
| POST   | `/api/history/save`   | Save analysis to history     |
| GET    | `/api/health`         | Health check                 |

---

## 🤝 Contributing

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
