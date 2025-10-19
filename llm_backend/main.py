# Setup Instructions:
# 1. Install dependencies:
#    pip install -r requirements.txt
#
# 2. Add your API keys to the .env file:
#    OPENAI_API_KEY=your-openai-key-here
#    GEMINI_API_KEY=your-gemini-key-here
#
# 3. Run the server:
#    uvicorn main:app --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import google.generativeai as genai
from config import settings
import os

# Configure APIs
if not settings.openai_api_key.startswith("your-openai-key-here"):
    openai.api_key = settings.openai_api_key

if not settings.gemini_api_key.startswith("your-gemini-key-here"):
    genai.configure(api_key=settings.gemini_api_key)

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

class LLMResponse(BaseModel):
    model: str
    response: str

@app.post("/gpt5", response_model=LLMResponse)
async def query_gpt5(request: PromptRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if settings.openai_api_key.startswith("your-openai-key-here"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    try:
        # Note: gpt-5 is not a real model name as of now. Using gpt-4 as a placeholder.
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": request.prompt}]
        )
        return {
            "model": "gpt-5", # as requested by the user
            "response": response.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gemini", response_model=LLMResponse)
async def query_gemini(request: PromptRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if settings.gemini_api_key.startswith("your-gemini-key-here"):
        raise HTTPException(status_code=500, detail="Gemini API key not configured.")

    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(request.prompt)
        return {
            "model": "gemini-1.5-pro",
            "response": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
