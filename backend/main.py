from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Mistral API URL and API key
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

class ChatRequest(BaseModel):
    message: str
    code: str = ""

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "mistral-large-latest",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant specializing in LeetCode problems."
                },
                {
                    "role": "user",
                    "content": f"Question: {request.message}\nCode: {request.code}"
                }
            ]
        }

        response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
        response.raise_for_status()

        chat_response = response.json()
        return ChatResponse(response=chat_response["choices"][0]["message"]["content"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Sample problems endpoint
@app.get("/api/problems")
async def get_problems():
    return {
        "problems": [
            {
                "id": 1,
                "title": "Two Sum",
                "difficulty": "Easy",
                "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target...",
                "examples": [
                    {
                        "input": "nums = [2,7,11,15], target = 9",
                        "output": "[0,1]"
                    }
                ]
            }
        ]
    }
