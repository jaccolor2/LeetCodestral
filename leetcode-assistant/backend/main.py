from fastapi import FastAPI, HTTPException, Response, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
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

class ChatResponse(BaseModel):
    response: str

class CodeExecutionRequest(BaseModel):
    code: str

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

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Check moderation first
        is_safe = await check_moderation(request.message)
        if not is_safe:
            def generate_moderation_message():
                yield json.dumps({
                    "role": "assistant",
                    "content": "Your message was flagged as inappropriate. Please rephrase your question.",
                    "timestamp": int(time.time() * 1000)
                }) + "\n"
                yield json.dumps({
                    "role": "assistant",
                    "content": "[DONE]",
                    "timestamp": int(time.time() * 1000)
                }) + "\n"

            return StreamingResponse(
                generate_moderation_message(),
                media_type="text/event-stream"
            )

        # Add debug logs
        print("Received chat request:")
        print(f"Message: {request.message}")
        print(f"Code length: {len(request.code)}")
        print(f"History length: {len(request.history)}")
        print(f"Test results: {request.testResults}")
        
        # Get problem details
        problems = await get_problems()
        problem = next((p for p in problems["problems"] if p["id"] == request.problem_id), None)
        if not problem:
            return JSONResponse(
                status_code=404,
                content={"detail": "Problem not found"}
            )

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

        def generate():
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

        # Save conversation to database
        conversation = Conversation(
            user_id=current_user.id,
            problem_id=request.problem_id,
            messages=request.history + [{"role": "user", "content": request.message}]
        )
        db.add(conversation)
        db.commit()

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(e)
            }
        )

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
        # Create problem generation prompt
        prompt = load_prompt("problem_generation")
        
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

        # Compare both prompts after formatting
        file_prompt = load_prompt("test_generation").format(
            problem_description=problem['description'],
            function_name=problem['functionName'],
            solution_code=request.code
        )
        
        hardcoded_prompt = TEST_GENERATION_PROMPT.format(
            problem_description=problem['description'],
            function_name=problem['functionName'],
            solution_code=request.code
        )
        
        print("File prompt after formatting:>>>\n", file_prompt, "\n<<<end")
        print("Hardcoded prompt after formatting:>>>\n", hardcoded_prompt, "\n<<<end")
        print("Are they different?", file_prompt != hardcoded_prompt)
        
        # Continue with the original code using file_prompt
        test_prompt = file_prompt

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
                        "content": "You are a Python test case generator. Generate test cases with VALID PYTHON SYNTAX. Also you should give the python code in a json object with the following format: { \"python_code\": pythoncode}"
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

        # Create a StringIO to capture print output
        output_buffer = StringIO()
        sys.stdout = output_buffer

        try:
            # Execute the test code using the function name from the problem
            exec_code = f"""
{request.code}

{test_cases['python_code']}

# Run the tests
test_{problem['functionName']}()
"""
            namespace = {}
            exec(exec_code, namespace)
            
            # Get the captured output
            output = output_buffer.getvalue()
            
            # Parse the output to get test results
            output_lines = output.strip().split('\n')
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
                    # Don't fail the whole request if validation fails
                    pass

            return {
                "results": results,
                "validation": validation_result.dict() if validation_result else None
            }

        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

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

async def check_moderation(content: str) -> bool:
    """
    Returns True if content is safe, False if it should be blocked
    """
    try:
        # Define blocked categories and their descriptions
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
        
        print("\n=== MODERATION CHECK ===")
        print(f"Content to moderate: {content}")

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

        print(f"\nModeration API Status Code: {moderation_response.status_code}")
        print(f"Full API Response: {moderation_response.text}")

        if moderation_response.status_code != 200:
            print(f"Moderation API error: {moderation_response.text}")
            return True  # Default to allowing if API fails

        results = moderation_response.json().get("results", [{}])[0]
        categories = results.get("categories", {})
        scores = results.get("category_scores", {})
        
        print("\nDetected Categories:")
        for category, is_detected in categories.items():
            score = scores.get(category, 0)
            print(f"- {category}: {is_detected} (score: {score:.4f})")
    
        for category in blocked_categories:
            if categories.get(category, False):
                print(f"\n❌ Content blocked due to {category} content")
                return False
        
        print("\n✅ Content passed moderation check")
        return True

    except Exception as e:
        print(f"\n⚠️ Moderation check failed with error: {e}")
        return True  # Default to allowing if check fails

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
