# Smart Prompt Router

A DistilBERT-based classifier to route lightweight and heavyweight prompts to different agents. This project includes a trained model, a FastAPI backend, and a web-based dashboard for real-time classification and routing.

## Project Structure

```
.
├── backend
│   ├── app.py
│   └── requirements.txt
├── data
│   └── prompts.csv
├── frontend
│   └── index.html
├── models
│   └── distilbert-prompt-classifier
│       ├── ... (model files)
├── scripts
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── config.yaml
├── gemini.md
└── requirements.txt
```

## Setup and Usage

### 1. Install Dependencies

There are two `requirements.txt` files. One for the core project (training scripts) and one for the backend.

**Core dependencies:**
```bash
pip install -r requirements.txt
```

**Backend dependencies:**
```bash
pip install -r backend/requirements.txt
```

### 2. Run the Backend

The backend is a FastAPI application that serves the trained model.

```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
*Note: Depending on your Python environment, you might need to use the full path to `uvicorn`.*

### 3. Use the Frontend

The frontend is a single HTML file that provides a dashboard to interact with the backend.

1.  Make sure the backend server is running.
2.  Open the `frontend/index.html` file in your web browser.

You can now enter prompts in the text area and see the classification results in real-time.