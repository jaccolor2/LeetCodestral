import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Mistral API URL and API key
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Test generation prompt template
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

def generate_tests(code: str, problem_id: int):
    try:
        # Mock problem details (normally from database)
        problem = {
            'id': 1,
            'description': 'Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.',
            'functionName': 'twoSum'
        }

        # Create test generation prompt
        test_prompt = TEST_GENERATION_PROMPT.format(
            problem_description=problem['description'],
            function_name=problem['functionName'],
            solution_code=code
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
        return content

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Test data
    code = """def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []"""

    # Generate and run tests
    test_cases = generate_tests(code, 1)
    print("\nGenerated Test Cases:")
    print(test_cases)

    if test_cases:
        try:
            # Parse the test cases
            test_cases = json.loads(test_cases)
            
            # Execute the test code
            exec_code = f"""
{code}

{test_cases['python_code']}

# Run the tests
test_twoSum()
"""
            print("\nExecuting tests:")
            namespace = {}
            exec(exec_code, namespace)

        except Exception as e:
            print(f"\nError executing tests: {e}") 