from fastapi import FastAPI, HTTPException, Response, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Set
import os
import requests
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse, JSONResponse
import json
import subprocess
import tempfile
import ast
import inspect
import time
from io import StringIO
import sys
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from database import get_db
from models import User, Conversation
from datetime import datetime, timedelta

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

# Add these near the top with other imports
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Add after other global variables
PROBLEM_TITLE_CACHE: Set[str] = set()

class TestResult(BaseModel):
    description: str
    input: str
    expected_output: str
    output: str
    passed: bool
    error: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    code: str = ""
    problem_id: int
    history: List[dict] = []
    testResults: Optional[List[TestResult]] = None
    problem: Optional[dict] = None  # Add current problem to request

class ChatResponse(BaseModel):
    response: str

class CodeExecutionRequest(BaseModel):
    code: str
    language: str  # Add language field

class CodeEvaluator:
    def __init__(self, problem):
        self.problem = problem

    def run_tests(self, code: str) -> List[TestResult]:
        # Basic implementation
        result = TestResult()
        try:
            # Execute the code in a safe environment
            # This is a basic implementation - you might want to add more security
            local_dict = {}
            exec(code, {}, local_dict)
            
            result.output = "Code executed successfully"
            result.passed = True
        except Exception as e:
            result.error = str(e)
            result.passed = False
        
        return [result]

def load_prompt(filename: str) -> str:
    with open(f"prompts/{filename}.txt", "r") as f:
        content = f.read()
        # Replace double backslashes with single backslashes
        return content.replace('\\\\n', '\n')

def format_prompt(question: str, code: str, history: list, problem: dict, test_results: Optional[List[dict]] = None) -> dict:
    system_prompt = load_prompt("chat").format(
        problem_title=problem['title'],
        problem_difficulty=problem['difficulty'],
        problem_description=problem['description'],
        examples=problem['examples'],
        test_results=test_results if test_results else "No test results available"
    )

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

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

def authenticate_user(email: str, password: str, db: Session):
    user = db.query(User).filter(User.email == email).first()
    if not user or not pwd_context.verify(password, user.hashed_password):
        return None
    return user

class ModerationFailureResponse(BaseModel):
    is_moderated: bool = True
    reason: str
    categories: List[str]

async def check_moderation(content: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Returns (is_safe, category, score) if unsafe, (True, None, None) if safe
    """
    try:
        blocked_categories = {
            "sexual": "sexual or adult content",
            "hate_and_discrimination": "hateful or discriminatory content",
            "violence_and_threats": "violent content or threats",
            "dangerous_and_criminal_content": "dangerous or illegal activities",
            "selfharm": "content related to self-harm",
            "health": "misleading health information",
            "financial": "misleading financial advice",
            "law": "misleading legal advice",
            "pii": "personal identifiable information"
        }
        
        moderation_response = requests.post(
            "https://api.mistral.ai/v1/chat/moderations",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "input": [{
                    "content": content,
                    "role": "user"
                }],
                "model": "mistral-moderation-latest"
            }
        )

        if moderation_response.status_code != 200:
            return True, None, None

        results = moderation_response.json().get("results", [{}])[0]
        categories = results.get("categories", {})
        scores = results.get("category_scores", {})
        
        for category in blocked_categories:
            if categories.get(category, False):
                score = scores.get(category, 0)
                return False, category, f"{score:.4f}"
        
        return True, None, None

    except Exception as e:
        print(f"\n⚠️ Moderation check failed with error: {e}")
        return True, None, None

async def generate_moderation_response(category: str, score: str) -> dict:
    """
    Generate a moderation response configuration to be used with the streaming function
    """
    try:
        prompt = f"""As a friendly coding assistant, explain to the user why their message was flagged for moderation.
Category: {category} (confidence: {score})

Be polite but firm. Explain why such content isn't appropriate in a coding learning environment.
Keep the response short and friendly. Encourage them to rephrase their question to focus on the coding problem.

Response:"""

        print("\n=== GENERATING MODERATION RESPONSE ===")
        print(f"Category: {category}")
        print(f"Score: {score}")
        print(f"Prompt: {prompt}")

        return {
            "model": "mistral-large-latest",
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": 0.7,
            "stream": True
        }

    except Exception as e:
        print(f"\n⚠️ Error generating moderation response: {e}")
        return None

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Use the provided problem if available
        problem = request.problem
        if not problem:
            # Fallback to fetching problem if not provided
            problems = await get_problems()
            if not problems or not problems.get("problems"):
                return StreamingResponse(
                    generate_stream(
                        MockResponse(
                            "I'm still loading the problem data. Please wait a moment and try again."
                        )
                    ),
                    media_type="text/event-stream"
                )
            problem = next((p for p in problems["problems"] if p["id"] == request.problem_id), None)
            if not problem:
                return StreamingResponse(
                    generate_stream(
                        MockResponse(
                            "I couldn't find that problem. Please try refreshing the page."
                        )
                    ),
                    media_type="text/event-stream"
                )

        def generate_stream(response):
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        try:
                            json_data = json.loads(decoded_line[6:])
                            if "choices" in json_data and len(json_data["choices"]) > 0:
                                content = json_data["choices"][0]["delta"].get("content", "")
                                if content:
                                    yield json.dumps({
                                        "role": "assistant",
                                        "content": content,
                                        "timestamp": int(time.time() * 1000)
                                    }) + "\n"
                        except json.JSONDecodeError:
                            continue
            yield json.dumps({
                "role": "assistant",
                "content": "[DONE]",
                "timestamp": int(time.time() * 1000)
            }) + "\n"

        # Add debug logs
        print("Received chat request:")
        print(f"Message: {request.message}")
        print(f"Code length: {len(request.code)}")
        print(f"History length: {len(request.history)}")
        print(f"Test results: {request.testResults}")
        
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        data = format_prompt(request.message, request.code, request.history, problem, request.testResults)
        data["stream"] = True
        data["temperature"] = 0.8

        # Log the formatted prompt
        formatted_prompt = data
        print("\nFormatted prompt sent to model:")
        print(formatted_prompt)

        # Make streaming request
        response = requests.post(MISTRAL_API_URL, headers=headers, json=data, stream=True)
        
        # Check for HTTP errors
        if response.status_code != 200:
            return JSONResponse(
                status_code=response.status_code,
                content={
                    "detail": f"Mistral API returned {response.status_code}: {response.text}"
                }
            )

        return StreamingResponse(generate_stream(response), media_type="text/event-stream")
    except Exception as e:
        print(f"Chat error: {e}")  # Add logging
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"An error occurred: {str(e)}"
            }
        )

@app.post("/api/execute")
async def execute_code(request: CodeExecutionRequest):
    try:
        # Create a temporary file with the appropriate extension
        suffix = '.js' if request.language == 'javascript' else '.py'
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(request.code)
            temp_file_path = f.name

        try:
            # Run the code with a timeout
            if request.language == 'javascript':
                process = subprocess.run(
                    ['node', temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=5  # 5 second timeout
                )
            else:  # Default to Python
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

@app.post("/api/evaluate")
async def evaluate_code(request: CodeExecutionRequest):
    try:
        # Get problem details
        problems = await get_problems()
        problem = next((p for p in problems["problems"] if p["id"] == request.problem_id), None)
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")

        # Run tests
        evaluator = CodeEvaluator(problem)
        test_results = evaluator.run_tests(request.code)
        
        # Prepare analysis prompt for Mistral
        analysis_prompt = f"""
Analyze the following code solution for the {problem['title']} problem:

Code:
```python
{request.code}
```

Test Results:
{[f"Test {i+1}: {'Passed' if r.passed else 'Failed'} (Expected: {r.expected_output}, Got: {r.output})" for i, r in enumerate(test_results)]}

Please provide a detailed analysis of:
1. Correctness: Does the code solve the problem correctly?
2. Time Complexity: What is the time complexity of this solution?
3. Space Complexity: What is the space complexity of this solution?
4. Code Style: Is the code well-written and following best practices?
5. Potential Improvements: What could be improved?

Keep the analysis concise but thorough."""

        # Get Mistral's analysis
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            MISTRAL_API_URL,
            headers=headers,
            json={
                "model": "mistral-large-latest",
                "messages": [
                    {"role": "assistant", "content": "You are a code review expert."},
                    {"role": "user", "content": analysis_prompt}
                ]
            }
        )
        
        analysis = response.json()["choices"][0]["message"]["content"]

        return {
            "testResults": [vars(result) for result in test_results],
            "analysis": analysis,
            "allTestsPassed": all(r.passed for r in test_results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CodeValidationRequest(BaseModel):
    code: str
    problem_id: int

@app.post("/api/validate")
async def validate_code(request: CodeValidationRequest):
    try:
        # Get problem details
        problems = await get_problems()
        problem = next((p for p in problems["problems"] if p["id"] == request.problem_id), None)
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")

        # First, get Mistral's analysis of the code structure
        analysis_prompt = load_prompt("validation").format(
            problem_title=problem['title'],
            problem_description=problem['description'],
            code=request.code
        )

        # Get Mistral's classification with extremely lenient criteria
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            MISTRAL_API_URL,
            headers=headers,
            json={
                "model": "mistral-large-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an extremely supportive code validator. Your primary goal is to encourage learning and boost confidence. Always lean towards marking solutions as CORRECT unless they are completely wrong."
                    },
                    {"role": "user", "content": analysis_prompt}
                ]
            }
        )
        
        analysis = response.json()["choices"][0]["message"]["content"]
        
        # Parse the classification
        classification = "INCORRECT"
        reason = "Could not determine"
        
        for line in analysis.split('\n'):
            if line.startswith('CLASSIFICATION:'):
                classification = line.split(':')[1].strip()
            elif line.startswith('REASON:'):
                reason = line.split(':')[1].strip()

        # Run basic tests if needed, but with very lenient criteria
        if classification == "CORRECT":
            evaluator = CodeEvaluator(problem)
            test_results = evaluator.run_tests(request.code)
            
            # Consider correct if code runs without errors, even if tests don't pass
            has_no_errors = all(not r.error for r in test_results)
            some_output = any(r.output for r in test_results)
            
            if has_no_errors and some_output:
                classification = "CORRECT"
                reason += "\n\nYour code runs successfully! Keep refining it to handle all test cases perfectly!"
            else:
                # Still encourage them even if there are errors
                classification = "INCORRECT"
                reason = "You're on the right track! Your logic looks good, just need to fix a few small issues. Keep going!"

        return {
            "classification": classification,
            "reason": reason,
            "testResults": [vars(result) for result in test_results] if test_results else None,
            "nextProblem": problem["id"] + 1 if classification == "CORRECT" else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/problems")
async def get_problems():
    try:
        # Create problem generation prompt with cached titles
        prompt = load_prompt("problem_generation")
        if PROBLEM_TITLE_CACHE:
            prompt += f"\n\nAvoid these already used titles: {', '.join(PROBLEM_TITLE_CACHE)}"
        
        # Make request to Mistral API with higher temperature for variety
        response = requests.post(
            MISTRAL_API_URL,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistral-large-latest",
                "temperature": 0.7,  # Increased temperature for more variety
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "response_format": {
                    "type": "json_object"
                }
            }
        )

        problem_json = response.json()["choices"][0]["message"]["content"]
        problem = json.loads(problem_json)
        
        # Add the new problem title to cache
        PROBLEM_TITLE_CACHE.add(problem['title'])
        
        print("problem:>>>\n", problem, "\n<<<end")
        
        # Return the generated problem in the expected format
        return {
            "problems": [problem]
        }

    except Exception as e:
        print("Error generating problem:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

# Add these new classes at the top with other models
class TestGenerationRequest(BaseModel):
    code: str
    problem_id: int

class TestCase(BaseModel):
    input: str
    expected_output: str
    description: str = ""

TEST_GENERATION_PROMPT = """Given this Python function and problem description, generate 3 test cases that thoroughly test the solution.

Problem Description: {problem_description}
Solution Code:
{solution_code}

Requirements:
1. Generate test cases that cover:
   - Normal cases
   - Edge cases
   - Common error scenarios
2. Make sure test cases are valid Python expressions
3. Ensure inputs match the function parameters
4. Expected outputs should match the function return type

example output (with ".." and "n" being respectively the inputs and function parameters):
{{
  "python_code": "def test_{function_name}():\\n    try:\\n        # Test case n: Description\\n        result = {function_name}([.., .., .., ..], ..)\\n        expected = [.., ..]\\n        print(f\\"Test case n: {{' Passed' if result == expected else ' Failed'}}\\")\n        print(f\\"  Input: nums=[.., .., .., ..], target=..\\")\n        print(f\\"  Expected: {{expected}}\\")\n        print(f\\"  Got: {{result}}\\")\n        assert result == expected\\n\\n        print(\\"\\\\nAll tests passed!\\")\n    except AssertionError as e:\\n        print(f\\"\\\\nTest failed: {{e}}\\")"
}}"""


# Add these new endpoints
class GenerateTestsRequest(BaseModel):
    code: str
    problem_id: int

@app.post("/api/generate-tests")
async def generate_tests(request: GenerateTestsRequest, problem=None):
    try:
        # Get problem details if not provided
        if not problem:
            problems = await get_problems()
            problem = next((p for p in problems["problems"] if p["id"] == request.problem_id), None)
            
            if not problem:
                raise HTTPException(status_code=404, detail="Problem not found")

        # Detect language based on code content
        language = "javascript" if "function" in request.code else "python"
        
        # Format the test generation prompt with language
        test_prompt = load_prompt("test_generation").format(
            language=language,
            problem_description=problem['description'],
            function_name=problem['functionName'],
            solution_code=request.code
        )

        # Make request to Mistral API
        response = requests.post(
            MISTRAL_API_URL,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistral-large-latest",
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "user",
                        "content": "You are a Python and JavaScript test case generator. Generate test cases with VALID SYNTAX for the specified language. Also you should give the code in a json object with the following format: { \"python_code\": pythoncode } or { \"javascript_code\": jscode }"
                    },
                    {
                        "role": "user",
                        "content": test_prompt
                    }
                ],
                "response_format": {
                    "type": "json_object"
                }
            }
        )

        print("response:>>>\n", response.json(), "\n<<<end")
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run-tests")
async def run_tests(request: GenerateTestsRequest):
    try:
        # Get problem details first
        problems = await get_problems()
        problem = next((p for p in problems["problems"] if p["id"] == request.problem_id), None)
        print(problem)
        
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")

        # Generate test cases with the problem
        test_cases = await generate_tests(request, problem)
        print("\nTest Cases:")
        print(test_cases)

        # Create a temporary file with the appropriate extension and content
        language = "javascript" if "function" in request.code else "python"
        suffix = '.js' if language == 'javascript' else '.py'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            # Write the solution code first
            f.write(request.code + "\n\n")
            
            # Write the test code
            if language == "javascript":
                f.write(test_cases.get("javascript_code", ""))
            else:
                f.write(test_cases.get("python_code", ""))
            
            temp_file_path = f.name

        try:
            # Execute the code with a timeout
            if language == "javascript":
                process = subprocess.run(
                    ['node', temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            else:
                process = subprocess.run(
                    ['python', temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

            # Parse the output to get test results
            output_lines = process.stdout.strip().split('\n')
            results = []
            current_test = {}
            
            for line in output_lines:
                if line.startswith("Test case"):
                    if current_test:
                        results.append(current_test)
                    current_test = {
                        "description": line,
                        "passed": "Passed" in line
                    }
                elif "Input:" in line:
                    current_test["input"] = line.split("Input:")[1].strip()
                elif "Expected:" in line:
                    current_test["expected_output"] = line.split("Expected:")[1].strip()
                elif "Got:" in line:
                    current_test["output"] = line.split("Got:")[1].strip()
            
            if current_test:
                results.append(current_test)

            # After getting the test results, check if all tests passed
            all_tests_passed = all(result.get("passed", False) for result in results)
            
            # If all tests passed, run validation
            validation_result = None
            if all_tests_passed:
                try:
                    validation_request = ValidationRequest(
                        code=request.code,
                        problem_id=request.problem_id
                    )
                    validation_result = await validate(validation_request)
                except Exception as e:
                    print(f"Validation error: {e}")
                    pass

            return {
                "results": results,
                "validation": validation_result.dict() if validation_result else None
            }

        except subprocess.TimeoutExpired:
            return {
                "results": [{
                    "description": "Test execution",
                    "passed": False,
                    "input": "N/A",
                    "expected_output": "N/A",
                    "output": "Code execution timed out after 5 seconds"
                }]
            }
        except Exception as e:
            return {
                "results": [{
                    "description": "Test execution",
                    "passed": False,
                    "input": "N/A",
                    "expected_output": "N/A",
                    "output": str(e)
                }]
            }
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add these new classes near the top with other BaseModel classes
class ValidationRequest(BaseModel):
    code: str
    problem_id: int

class ValidationResponse(BaseModel):
    classification: str
    reason: str
    test_results: Optional[List[dict]] = None
    next_problem: Optional[int] = None

# Add this new endpoint
@app.post("/api/validate")
async def validate(request: ValidationRequest):
    try:
        # Get problem details
        problems = await get_problems()
        problem = next((p for p in problems["problems"] if p["id"] == request.problem_id), None)
        
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")

        # Load validation prompt
        validation_prompt = load_prompt("validation").format(
            problem_title=problem['title'],
            problem_description=problem['description'],
            code=request.code
        )

        # Call Mistral API for validation
        response = requests.post(
            MISTRAL_API_URL,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistral-large-latest",
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "user",
                        "content": validation_prompt
                    }
                ]
            }
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to validate code")

        # Parse the response
        content = response.json()["choices"][0]["message"]["content"]
        
        # Extract classification and reason with explicit mapping
        classification = "INCORRECT"
        reason = "Could not validate the solution"
        
        for line in content.split('\n'):
            if line.startswith("CLASSIFICATION:"):
                raw_classification = line.split(':')[1].strip().upper()
                # Ensure classification is exactly 'CORRECT' or 'INCORRECT'
                classification = 'CORRECT' if raw_classification == 'CORRECT' else 'INCORRECT'
            elif line.startswith("REASON:"):
                reason = line.split(':')[1].strip()

        print(f"Validation result: classification={classification}, reason={reason}")  # Debug log

        # Prepare the response
        validation_response = ValidationResponse(
            classification=classification,
            reason=reason,
            next_problem=problem['id'] + 1 if classification == 'CORRECT' else None
        )

        print(f"Sending validation response: {validation_response.dict()}")  # Debug log
        return validation_response

    except Exception as e:
        print(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this with the other models at the top of the file
class UserCreate(BaseModel):
    email: str
    password: str

# Update the register endpoint
@app.post("/api/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create new user with hashed password
    hashed_password = pwd_context.hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    
    # Return JWT token
    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

class ModerationRequest(BaseModel):
    input: List[Dict[str, str]]
    model: str = "mistral-large-latest"

@app.post("/api/moderate")
async def moderate_content(request: ModerationRequest):
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/moderations",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            },
            json=request.dict()
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Moderation API error: {response.text}"
            )
            
        return response.json()
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Moderation check failed: {str(e)}"
        )
# Add this helper class at the top of the file
class MockResponse:
    def __init__(self, message: str):
        self.message = message

    def iter_lines(self):
        yield json.dumps({
            "choices": [{
                "delta": {
                    "content": self.message
                }
            }]
        }).encode()

