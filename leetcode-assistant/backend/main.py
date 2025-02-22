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

class UserProgress(BaseModel):
    code_attempts: List[str]
    test_results: List[dict]
    current_problem: dict
    validation_history: List[dict]

user_progress = {} 

class CodeEvaluator:
    def __init__(self, problem: dict):
        self.problem = problem
        self.function_name = problem['functionName']
        
    def prepare_test_code(self, user_code: str, test_case: dict) -> str:
        # Wrap the user's code with our test harness
        test_wrapper = f"""
{user_code}

# Test execution
def run_test():
    try:
        input_values = {test_case['input']}
        result = {self.function_name}(*input_values)
        return str(result)
    except Exception as e:
        return f"Error: {{str(e)}}"

print(run_test())
"""
        return test_wrapper

    def run_tests(self, code: str) -> List[TestResult]:
        results = []
        for test_case in self.problem['testCases']:
            try:
                # Create a temporary file with the test code
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    test_code = self.prepare_test_code(code, test_case)
                    f.write(test_code)
                    temp_file_path = f.name

                # Run the test
                process = subprocess.run(
                    ['python', temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                # Compare output with expected
                actual_output = process.stdout.strip()
                expected_output = str(test_case['expected_output'])
                passed = actual_output == expected_output

                results.append(TestResult(
                    passed=passed,
                    output=actual_output,
                    expected=expected_output,
                    error=process.stderr if process.stderr else None
                ))

            except Exception as e:
                results.append(TestResult(
                    passed=False,
                    output="",
                    expected=str(test_case['expected_output']),
                    error=str(e)
                ))
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        return results

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
        analysis_prompt = f"""
You are an extremely encouraging code validator for beginners. Your goal is to boost confidence and motivation.
Analyze this solution for the {problem['title']} problem with a very lenient approach.

Problem: {problem['description']}

Proposed solution:
```python
{request.code}
```

Evaluation Guidelines:
- If the code shows ANY attempt to solve the problem, consider it CORRECT
- If they use relevant concepts (like loops, if statements, etc.), mark it CORRECT
- If they show understanding of the basic problem, even if implementation is incomplete, mark it CORRECT
- Only mark as INCORRECT if the code is completely unrelated or empty

Default to CORRECT unless there's a strong reason not to.
Always provide very encouraging feedback that motivates further learning.

Only respond in this format:
CLASSIFICATION: [CORRECT/INCORRECT]
REASON: [Your very encouraging explanation highlighting what they did well]
"""

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
        problem_prompt = """Generate a simple coding problem similar to LeetCode problems. The problem should be beginner-friendly.

IMPORTANT: Respond ONLY with a valid JSON object in exactly this format, with no additional text or formatting:
{
    "id": 1,
    "title": "A clear, concise title",
    "difficulty": "Easy",
    "description": "A clear problem description with specific input/output requirements",
    "functionName": "camelCaseFunctionName",
    "parameters": ["parameter1", "parameter2"],
    "examples": [
        {
            "input": "Simple input as string",
            "expected_output": "Expected output as string",
            "explanation": "Clear explanation of this example"
        }
    ],
    "testCases": [
        {"input": "Test input 1", "expected_output": "Expected output 1"},
        {"input": "Test input 2", "expected_output": "Expected output 2"},
        {"input": "Test input 3", "expected_output": "Expected output 3"}
    ]
}

Make sure:
- All JSON is properly formatted with double quotes
- All values are strings (except id)
- No trailing commas
- No comments or additional text
- Function name is in camelCase
- Examples are simple and clear
- Test cases match the problem requirements"""

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
                ]
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
                        "functionName": "twoSum",
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



