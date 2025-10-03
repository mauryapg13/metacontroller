from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and tokenizer
model_path = "../models/distilbert-prompt-classifier/"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

class PromptRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Smart Prompt Router API is running"}

@app.post("/classify")
def classify_prompt(request: PromptRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = F.softmax(logits, dim=1).squeeze()
    predicted_class_id = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class_id].item()

    label_map = {0: "light", 1: "heavy"}
    predicted_label = label_map[predicted_class_id]

    return {"label": predicted_label, "confidence": confidence}
