from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from .model_loader import model
from .utils import preprocess_image, preprocess_text, get_disease_info

app = FastAPI(title="AI Skin Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = sorted([
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis",
    "Vascular lesion"
])

@app.get("/")
def home():
    return {"message": "Skin AI Backend Running Successfully"}

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    symptoms: str = Form(...)
):
    try:
        contents = await file.read()

        img = preprocess_image(contents)
        text_input = preprocess_text(symptoms)

        prediction = model.predict([img, text_input])

        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        disease_name = CLASS_NAMES[predicted_index]
        info = get_disease_info(disease_name)

        # -----------------------------
        # RISK & SEVERITY LOGIC
        # -----------------------------
        doctor_required = False
        risk_level = "Low"
        consultation_message = "No immediate consultation required."
        severity_color = "green"

        if info:
            if info["deadly"].lower() == "yes":
                doctor_required = True
                risk_level = "High"
                severity_color = "red"
                consultation_message = (
                    "This condition may be serious. Immediate consultation with a dermatologist is strongly recommended."
                )

            elif confidence < 70:
                doctor_required = True
                risk_level = "Moderate"
                severity_color = "yellow"
                consultation_message = (
                    "Prediction confidence is low. It is advised to consult a dermatologist for accurate diagnosis."
                )

        return {
            "disease": disease_name,
            "confidence": round(confidence, 2),
            "risk_level": risk_level,
            "severity_color": severity_color,
            "doctor_consultation_required": doctor_required,
            "consultation_message": consultation_message,
            "reason": info["reason"] if info else "",
            "precaution": info["precaution"] if info else "",
            "deadly": info["deadly"] if info else "",
            "contagious": info["contagious"] if info else "",
            "cure": info["cure"] if info else ""
        }

    except Exception as e:
        return {"error": str(e)}
