import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add this after load_dotenv()
print("Debug - Current directory:", os.getcwd())
print("Debug - .env exists:", os.path.exists('.env'))

# Get API key from environment and debug
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
print("Debug - API Key:", MISTRAL_API_KEY)  # Add this to check the key
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment variables")

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def test_generate_tests():
    # Test generation prompt
    test_prompt = """Given this Python function and problem description, generate 3 test cases that thoroughly test the solution.

Problem Description: Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

Function Signature:
def twoSum(nums, target):
    pass

Requirements:
1. Generate test cases that cover:
   - Normal cases
   - Edge cases
   - Common error scenarios
2. Make sure test cases are valid Python expressions
3. Ensure inputs match the function parameters
4. Expected outputs should match the function return type

example output (with ".." and "n" being respectively the inputs and function parameters):
{
  "python_code": "def test_twoSum():\n    # Test case n: Normal case\n    assert twoSum([.., .., .., ..], ..) == [.., ..]\n   }
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
                "type": "json_object",
            }
        }
    )

    # Print raw response
    print("Raw Response:")
    print(json.dumps(response.json(), indent=2))

    # Get message content
    try:
        content = response.json()["choices"][0]["message"]["content"]
        print("\nGenerated Test Cases:")
        print(content)
    except Exception as e:
        print(f"\nError parsing response: {e}")

if __name__ == "__main__":
    test_generate_tests() 