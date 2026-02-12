# 🧠 AI Skin Disease Detection System

An AI-powered web application that detects skin diseases using deep learning.
Users can upload an image of a skin condition and optionally describe symptoms.
The system processes the image using a trained TensorFlow model and returns predictions with confidence scores and medical recommendations.

---

## 📌 1. Project Overview

### 🔎 Problem Statement

Skin diseases are common worldwide, but early diagnosis is often delayed due to:

- Limited access to dermatologists
- High consultation costs
- Lack of awareness
- Misinterpretation of symptoms

Delayed diagnosis can lead to severe complications, especially in cases like melanoma.

### 💡 Solution

This system provides:

- **AI-based skin disease detection**
- **Image + symptom-based prediction** (Multimodal model)
- **Confidence score output**
- **Doctor consultation alerts**
- **Cloud deployment for public accessibility**

It bridges the gap between AI and accessible healthcare.

---

## 🚀 2. Tech Stack

### Backend

- FastAPI
- Uvicorn
- Python 3.10

### Frontend

- React (Create React App)
- Hosted on Vercel

### Machine Learning

- TensorFlow 2.15 (Keras)
- Custom trained multimodal model (.keras)

### Image Processing

- Pillow
- NumPy

### Deployment

- Render → Backend
- Vercel → Frontend

---

## 🏗️ 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │    Frontend      │         │   Deployment     │         │
│  │  ┌────────────┐  │         │   (Vercel)       │         │
│  │  │   React    │  │         └──────────────────┘         │
│  │  │   (CRA)    │  │                                       │
│  │  └────────────┘  │              ↑                       │
│  │  ┌────────────┐  │              │                       │
│  │  │ Upload     │  │              │ HTTP / REST           │
│  │  │ Predict    │  │              │ (CORS Enabled)        │
│  │  │ Display    │  │              ↓                       │
│  └──────────────────┘         ┌──────────────────┐         │
│                                │     Backend      │         │
│                                │   ┌────────────┐ │         │
│                                │   │  FastAPI   │ │         │
│                                │   │  Uvicorn   │ │         │
│                                │   └────────────┘ │         │
│                                │   ┌────────────┐ │         │
│                                │   │ TensorFlow │ │         │
│                                │   │   2.15     │ │         │
│                                │   │  (Keras)   │ │         │
│                                │   └────────────┘ │         │
│                                │   ┌────────────┐ │         │
│                                │   │  Pillow    │ │         │
│                                │   │  NumPy     │ │         │
│                                │   └────────────┘ │         │
│                                │   (Render)       │         │
│                                └──────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 4. Project Structure

```
Skin-Disease-Detection-System/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── model_loader.py
│   │   ├── utils.py
│   │   └── disease_info.txt
│   │
│   ├── model/
│   │   └── multimodal_model.keras
│   │
│   ├── requirements.txt
│   └── .python-version
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   ├── package.json
│   └── package-lock.json
│
└── README.md
```

---

## 🛠️ 5. Prerequisites & Installation

### Requirements

- Python 3.10
- Node.js (18+ recommended)
- npm

### 🔹 Backend Setup

```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

pip install -r requirements.txt
```

Start backend:

```bash
uvicorn app.main:app --reload
```

Runs on:  
http://localhost:8000

### 🔹 Frontend Setup

```bash
cd frontend
npm install
npm start
```

Runs on:  
http://localhost:3000

---

## ⚙️ 6. Configuration

### Backend Environment Variables (`.env` example)

```
MODEL_PATH=model/multimodal_model.keras
ALLOWED_ORIGINS=http://localhost:3000
```

### Frontend Environment Variable

Create `.env` inside the `frontend/` directory:

```
REACT_APP_API_URL=http://localhost:8000
```

---

## 🚀 7. Deployment Guide

### 🔹 Backend Deployment (Render)

- **Root Directory:** `backend`
- **Build Command:**  
  `pip install -r requirements.txt`
- **Start Command:**  
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Health Check Endpoint:** `/health`

**Important:**
- Use Python 3.10
- Use TensorFlow 2.15
- Keep model inside `backend/model/`

### 🔹 Frontend Deployment (Vercel)

- **Root Directory:** `frontend`
- **Framework:** Create React App
- **Build Command:**  
  `npm run build`
- **Environment Variable:**  
  `REACT_APP_API_URL=https://your-render-backend-url.onrender.com`

---

## 📡 8. API Documentation

### Health Check

- **GET** `/health`

**Response:**
```json
{
  "status": "healthy"
}
```

---

### Prediction Endpoint

- **POST** `/predict`

**Request:**
- `image` (file upload)
- `symptoms` (optional text)

**Response:**
```json
{
  "prediction": "Melanoma",
  "confidence": 95.4,
  "consult_doctor": true,
  "message": "Immediate consultation recommended."
}
```

---

## 🎯 9. Features

- Real-time skin disease detection
- Image upload and preprocessing
- TensorFlow deep learning inference
- Multimodal (Image + Symptoms)
- Confidence score display
- Doctor consultation alert logic
- Modern premium UI
- Cloud deployment ready

---

## 🔧 10. Troubleshooting

- **CORS Errors**  
  Enable CORS middleware in FastAPI.

- **TensorFlow Installation Issues**  
  Ensure Python 3.10 is used  
  Use TensorFlow 2.15 only

- **Model Loading Errors**  
  Verify model path  
  Ensure model file exists in `backend/model/`

- **Port Conflict**  
  Change port:  
  `uvicorn app.main:app --port 8001`

---

## 📄 11. License

MIT License
