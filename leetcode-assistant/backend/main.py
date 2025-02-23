from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
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

class TestResult:
    def __init__(self, passed: bool, output: str, expected: str, error: str = None):
        self.passed = passed
        self.output = output
        self.expected = expected
        self.error = error



def load_prompt(filename: str) -> str:
    with open(f"prompts/{filename}.txt", "r") as f:
        return f.read()

def format_prompt(question: str, code: str, history: list, problem: dict) -> dict:
    system_prompt = load_prompt("chat").format(
        problem_title=problem['title'],
        problem_difficulty=problem['difficulty'],
        problem_description=problem['description'],
        examples=problem['examples']
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

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
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

        data = format_prompt(request.message, request.code, [], problem)
        data["stream"] = True

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
{[f"Test {i+1}: {'Passed' if r.passed else 'Failed'} (Expected: {r.expected}, Got: {r.output})" for i, r in enumerate(test_results)]}

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
        # Prompt for Mistral to generate a coding problem
        problem_prompt = load_prompt("problem_generation")

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
                        "content": "You are a JSON generator that creates coding problems. You MUST respond with ONLY a valid JSON object, no markdown, no explanations, no additional text."
                    },
                    {"role": "user", "content": problem_prompt}                    
                ],
                "response_format": {
                    "type": "json_object"
                }
            }
        )
        
        problem_json = response.json()["choices"][0]["message"]["content"]
        
        # Clean and parse the response
        try:
            # Remove any potential markdown formatting
            problem_json = problem_json.replace("```json", "").replace("```", "").strip()
            
            # Find the JSON object
            json_start = problem_json.find("{")
            json_end = problem_json.rfind("}") + 1
            
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No valid JSON object found in response")
                
            problem_json = problem_json[json_start:json_end]
            
            # Parse the JSON
            problem = json.loads(problem_json)
            
            # Validate required fields
            required_fields = ["id", "title", "difficulty", "description", "functionName", "parameters", "examples", "testCases"]
            missing_fields = [field for field in required_fields if field not in problem]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
            # Return in the expected format
            return {
                "problems": [problem]
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            print("Error parsing Mistral response:", e)
            print("Raw response:", problem_json)
            # Fallback to default problem
            return {
                "problems": [
                    {
                        "id": 1,
                        "title": "Two Sum",
                        "difficulty": "Easy",
                        "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice.",
                        "functionName": "two_sum",
                        "parameters": ["nums", "target"],
                        "examples": [
                            {
                                "input": "[2,7,11,15], 9",
                                "expected_output": "[0,1]",
                                "explanation": "Because nums[0] + nums[1] == 9, we return [0, 1]"
                            }
                        ],
                        "testCases": [
                            {"input": "[2,7,11,15], 9", "expected_output": "[0,1]"},
                            {"input": "[3,2,4], 6", "expected_output": "[1,2]"},
                            {"input": "[3,3], 6", "expected_output": "[0,1]"}
                        ]
                    }
                ]
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
async def generate_tests(request: GenerateTestsRequest):
    try:
        # Mock problem details (normally from database)
        problem = {
            'id': 1,
            'description': 'Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.',
            'functionName': 'two_sum'
        }

        # Create test generation prompt
        test_prompt = TEST_GENERATION_PROMPT.format(
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

        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run-tests")
async def run_tests(request: GenerateTestsRequest):
    try:
        # Generate test cases
        test_cases = await generate_tests(request)
        print("\nTest Cases:")
        print(test_cases)

        # Create a StringIO to capture print output
        output_buffer = StringIO()
        sys.stdout = output_buffer

        try:
            # Execute the test code
            exec_code = f"""
{request.code}

{test_cases['python_code']}

# Run the tests
test_two_sum()
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
                
            return {"results": results}

        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
