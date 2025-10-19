from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import threading
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading State ---
class ModelStatus:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loading = False
        self.error = None

model_status = ModelStatus()

def load_model():
    """Loads the model and tokenizer."""
    global model_status
    model_status.is_loading = True
    logger.info("Starting model loading...")
    try:
        # Construct absolute path to the model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "..", "models", "distilbert-prompt-classifier")
        logger.info(f"Loading model from: {model_path}")

        model_status.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        model_status.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        model_status.error = str(e)
    finally:
        model_status.is_loading = False

@app.on_event("startup")
async def startup_event():
    """Run model loading in a background thread."""
    thread = threading.Thread(target=load_model)
    thread.start()

# --- API Endpoints ---
class PromptRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Smart Prompt Router API is running"}

@app.get("/health")
def health_check():
    """Provides the status of the model loading."""
    if model_status.is_loading:
        return {"status": "loading_model"}
    if model_status.error:
        return {"status": "error", "detail": model_status.error}
    if model_status.model and model_status.tokenizer:
        return {"status": "ready"}
    return {"status": "initializing"}


@app.post("/classify")
def classify_prompt(request: PromptRequest):
    """Classifies the prompt as 'light' or 'heavy'."""
    if model_status.is_loading:
        raise HTTPException(status_code=503, detail="Model is still loading. Please try again in a moment.")
    if not model_status.model or not model_status.tokenizer:
        raise HTTPException(status_code=500, detail=f"Model not available. Error: {model_status.error}")

    try:
        inputs = model_status.tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model_status.model(**inputs).logits

        probabilities = F.softmax(logits, dim=1).squeeze()
        predicted_class_id = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_id].item()

        label_map = {0: "light", 1: "heavy"}
        predicted_label = label_map[predicted_class_id]

        return {"label": predicted_label, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error during classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during classification.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
