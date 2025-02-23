import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def test_run_tests():
    # Sample solution code for twoSum
    solution_code = """
def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
"""

    # Test generation prompt
    test_prompt = f"""Given this Python function and problem description, generate 3 test cases that thoroughly test the solution.

Problem Description: Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

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
  "python_code": "def test_twoSum():\\n    try:\\n        # Test case n: Description\\n        result = twoSum([.., .., .., ..], ..)\\n        expected = [.., ..]\\n        print(f\\"Test case n: {{' Passed' if result == expected else ' Failed'}}\\")\n        print(f\\"  Input: nums=[.., .., .., ..], target=..\\")\n        print(f\\"  Expected: {{expected}}\\")\n        print(f\\"  Got: {{result}}\\")\n        assert result == expected\\n\\n        print(\\"\\\\nAll tests passed!\\")\n    except AssertionError as e:\\n        print(f\\"\\\\nTest failed: {{e}}\\")"
}}
"""

    # Make request to Mistral API
    response = requests.post(
        MISTRAL_API_URL,
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistral-large-latest",
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

    try:
        # Get test cases from Mistral
        content = response.json()["choices"][0]["message"]["content"]
        

        # Create complete test code with print statements
        test_code = f"""
{solution_code}

{content['python_code']}

# Run the tests
test_twoSum()
"""
        print("\nExecuting tests:")
        
        # Execute the test code in a namespace
        namespace = {}
        exec(test_code, namespace)

    except AssertionError as e:
        print(f"\nTest failed: {e}")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    test_run_tests()