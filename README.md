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
The dependencies have been pinned to specific versions to ensure compatibility.

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
You can run it directly with Python:

```bash
python3 backend/app.py
```

The server will start on `http://0.0.0.0:8000`.
The first time you run it, it will take some time to download and cache the model.

You can check the status of the model loading by accessing the `/health` endpoint: `http://localhost:8000/health`

### 3. Use the Frontend

The frontend is a single HTML file that provides a dashboard to interact with the backend.

1.  Make sure the backend server is running.
2.  Open the `frontend/index.html` file in your web browser.

You can now enter prompts in the text area and see the classification results in real-time.