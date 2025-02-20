from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import requests
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import json
import subprocess
import tempfile

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
    problem_id: int

class ChatResponse(BaseModel):
    response: str

class CodeExecutionRequest(BaseModel):
    code: str

def format_prompt(question: str, code: str, history: list, problem: dict) -> dict:
    system_prompt = f"""You are a friendly and approachable coding assistant helping with LeetCode problems. Your personality is:
- Supportive and encouraging
- Casual but professional
- Clear and concise
- Interactive and engaging

Current Problem: {problem['title']} ({problem['difficulty']})
Problem Description: {problem['description']}
Examples: {problem['examples']}

When responding:
Reference function names when relevant
Be encouraging and positive
Keep explanations brief
Ask questions to guide thinking
Focus on one point at a time
Never provide complete solutions

Example responses:
1. "Nice update! 🎉 I see you added a loop to your twoSum function. For Two Sum, we need to ensure we're checking all pairs efficiently."
2. "No worries! 😊 In your findIndices function, let's think about how we can use a hash map to store the numbers we've seen."
3. "Awesome progress! 🚀 Your calculateSum function is looking good. Now we can think about optimizing the space complexity."
4. "Looking at your main function, I notice you're using nested loops. How about we explore a more efficient approach?"

Remember: Track code changes, reference function names, and vary your responses based on conversation context! Always reference the current problem."""

    # Build conversation history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add previous messages
    for msg in history:
        messages.append({
            "role": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"]
        })
    
    # Add current message
    user_content = f"""Here's the code I'm working with:

```{code}
```My question is: {question}

Please guide me to solve this problem without providing the complete solution."""

    messages.append({"role": "user", "content": user_content})

    return {
        "model": "mistral-large-latest",
        "messages": messages
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Get problem details
        problems = await get_problems()
        problem = next((p for p in problems["problems"] if p["id"] == request.problem_id), None)
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")

        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        data = format_prompt(request.message, request.code, [], problem)
        data["stream"] = True  # Enable streaming

        # Make streaming request
        response = requests.post(MISTRAL_API_URL, headers=headers, json=data, stream=True)
        response.raise_for_status()

        def generate():
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        try:
                            # Parse the JSON object
                            json_data = json.loads(decoded_line[6:])
                            # Extract and yield only the content
                            if "choices" in json_data and len(json_data["choices"]) > 0:
                                content = json_data["choices"][0]["delta"].get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
            yield "[DONE]"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/execute")
async def execute_code(request: CodeExecutionRequest):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(request.code)
            temp_file_path = f.name

        try:
            # Run the code with a timeout
            process = subprocess.run(
                ['python', temp_file_path],
                capture_output=True,
                text=True,
                timeout=5  # 5 second timeout
            )
            
            return {
                "stdout": process.stdout,
                "stderr": process.stderr,
                "error": None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "",
                "error": "Code execution timed out after 5 seconds"
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "error": "Error executing code"
            }
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
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



