<div align="center">

# Career Intelligence System

### AI-Powered Resume Scoring, Role Prediction, and Skill Gap Analysis Platform

An intelligent machine learning and NLP platform designed to parse resumes, classify target career pathways with confidence scores, benchmark competencies against job descriptions, and deliver explainable skill gap insights.

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Database-MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-spaCy-09A3D5?style=flat-square&logo=spacy&logoColor=white)](https://spacy.io/)
[![Security](https://img.shields.io/badge/Auth-JWT%20Bearer-000000?style=flat-square&logo=jsonwebtokens&logoColor=white)](https://jwt.io/)
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=flat-square)](LICENSE)

</div>

---

## Overview

The Career Intelligence System provides candidates, recruiters, and engineering students with actionable intelligence regarding career readiness. By combining Natural Language Processing (NLP) with trained classification models, the system processes unstructured resume documents, identifies technical domain proficiencies, and highlights missing capabilities required for specific technical roles.

---

## Core Capabilities

- **Intelligent Resume Parsing**: Ingests resume documents (PDF, DOCX, TXT) to extract technical skills, domain vocabularies, educational history, and professional achievements.
- **Role Suitability Prediction**: Multi-class machine learning classifiers evaluate extracted features and output target job role predictions accompanied by calibrated probability scores.
- **Job Description (JD) Gap Benchmark**: Dynamically cross-references parsed candidate profiles against specific job descriptions to surface required competencies, missing keywords, and match percentages.
- **Explainable AI (XAI)**: Generates feature attribution insights demonstrating the exact keywords and skills driving role classification.
- **User Authentication & Session Audit**: FastAPI backend implements JWT authentication (PBKDF2 password hashing) with MongoDB persistence to store and audit candidate evaluation histories.
- **Interactive UI**: Built with Streamlit to provide real-time dashboard analytics, interactive charts, and PDF download summaries.

---

## Architecture and System Design

```
[ Candidate Resume / JD ]
           │
           ▼
┌─────────────────────────┐
│     Streamlit UI        │  ──> User Interface & Interactive Dashboards
└──────────┬──────────────┘
           │
     REST API (HTTP)
           │
           ▼
┌─────────────────────────┐
│     FastAPI Backend     │  ──> Authentication (JWT), Session History & Validation
└──────────┬──────────────┘
           │
           ├──────────────────────────┐
           ▼                          ▼
┌───────────────────────┐  ┌───────────────────────┐
│   MongoDB Database    │  │   ML & NLP Engine     │
│   (User & Records)    │  │   - spaCy / NLTK      │
└───────────────────────┘  │   - scikit-learn      │
                           │   - Random Forest     │
                           │   - SHAP/LIME XAI     │
                           └───────────────────────┘
```

- **Frontend Application (`frontend/`)**: Streamlit application handling file uploads, client-side NLP tokenization, domain matching, inference orchestration, and visualization.
- **Backend Service (`backend/`)**: Asynchronous FastAPI service handling JWT authentication, password resets, health monitoring, and persistent storage in MongoDB.
- **Training Pipeline (`scripts/`)**: Jupyter notebooks and data preprocessing scripts for training multi-class role prediction classifiers.

---

## Tech Stack

| Domain | Technology |
|:---|:---|
| **Frontend Framework** | Streamlit |
| **Backend Framework** | FastAPI (Uvicorn ASGI) |
| **Programming Language**| Python 3.9+ |
| **Database** | MongoDB (Motor async driver / PyMongo) |
| **Machine Learning** | scikit-learn, Random Forest Classifier |
| **Natural Language Processing** | spaCy, NLTK, Regular Expression tokenizers |
| **Security & Authentication** | JWT (PyJWT), PBKDF2-SHA256 (Passlib), HTTPBearer |
| **Containerization** | Docker, Dockerfile per service |

---

## Project Structure

```
career_intelligence_system/
├── backend/                  # FastAPI REST API Service
│   ├── api/
│   │   └── routes.py         # Auth, record persistence, and health endpoints
│   ├── core/
│   │   └── database.py       # Motor async MongoDB connection pooling
│   ├── models/
│   │   └── schemas.py        # Pydantic request and response schemas
│   ├── main.py               # Application entry point (FastAPI + CORS + Uvicorn)
│   ├── requirements.txt      # Backend dependencies
│   └── Dockerfile            # Container configuration for backend
│
├── frontend/                 # Streamlit Web Application
│   ├── src/
│   │   ├── predictor.py      # Model loading and role prediction inference
│   │   ├── nlp_processor.py  # Text cleaning, extraction, and normalization
│   │   ├── skill_analyzer.py # Competency comparison and gap detection
│   │   ├── jd_analyzer.py    # Job description keyword and requirement parser
│   │   ├── domain_matcher.py # Domain classification and scoring logic
│   │   └── explainer.py      # Feature importance and prediction explanation
│   ├── app.py                # Streamlit entry point and dashboard views
│   ├── api_client.py         # Client abstraction for FastAPI communication
│   ├── requirements.txt      # Frontend dependencies
│   └── Dockerfile            # Container configuration for frontend
│
└── scripts/                  # Model Training and Dataset Utilities
    ├── train_models.ipynb    # Model training, hyperparameter tuning, and export
    └── data/                 # Raw and processed training datasets
```

---

## Getting Started

### Prerequisites

- Python 3.9 to 3.11
- MongoDB (Local instance or MongoDB Atlas connection string)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/KrishnaP1504/career_intelligence_system.git
cd career_intelligence_system
```

### 2. Backend Setup

Open a terminal in the `backend/` directory:

```bash
cd backend
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```properties
MONGO_URI=mongodb://localhost:27017
DB_NAME=career_intelligence_db
JWT_SECRET_KEY=your_secure_random_jwt_secret_key
```

Run the backend server:

```bash
python main.py
```

The FastAPI backend runs at `http://localhost:8000` with Swagger documentation available at `http://localhost:8000/docs`.

### 3. Frontend Setup

Open a second terminal in the `frontend/` directory:

```bash
cd frontend
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The frontend dashboard will be available at `http://localhost:8501`.

---

## REST API Reference

| Method | Endpoint | Description | Auth Required |
|:---|:---|:---|:---|
| **GET** | `/api/health` | Health check probe | No |
| **POST** | `/api/signup` | Register new user account | No |
| **POST** | `/api/login` | Authenticate user & return JWT | No |
| **POST** | `/api/forgot-password` | Request password reset token | No |
| **POST** | `/api/reset-password` | Reset password using reset token | No |
| **POST** | `/api/records` | Save resume analysis record | Yes (Bearer) |
| **GET** | `/api/records` | Retrieve candidate analysis history | Yes (Bearer) |
| **DELETE** | `/api/records/{id}` | Remove saved analysis record | Yes (Bearer) |

---

## Docker Deployment

Run both services via Docker:

```bash
# Build and run backend container
cd backend
docker build -t career-intel-backend .
docker run -p 8000:8000 -e MONGO_URI=your_mongo_uri career-intel-backend

# Build and run frontend container
cd ../frontend
docker build -t career-intel-frontend .
docker run -p 8501:8501 career-intel-frontend
```

---

## Author

**Krishnakumar Pipaliya**  
Mobile Application Developer & Machine Learning Enthusiast

- Portfolio: [krishnapipaliya.in](https://krishnapipaliya.in)
- LinkedIn: [linkedin.com/in/krishn-pipaliya1504](https://www.linkedin.com/in/krishn-pipaliya1504/)
- GitHub: [github.com/KrishnaP1504](https://github.com/KrishnaP1504)
- Email: [pipaliyakrishna1504@gmail.com](mailto:pipaliyakrishna1504@gmail.com)

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
