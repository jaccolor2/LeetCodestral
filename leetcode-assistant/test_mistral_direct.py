import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Mistral API URL and API key
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def load_prompt(filename: str) -> str:
    # Fix the path to look in backend/prompts
    prompt_path = os.path.join("backend", "prompts", f"{filename}.txt")
    with open(prompt_path, "r") as f:
        return f.read()

# Rest of the code stays the same... 