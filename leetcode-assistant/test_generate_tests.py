import requests
import json

def test_generate_tests():
    # Test data
    code = """def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []"""

    # Make request to the endpoint
    response = requests.post(
        "http://localhost:8000/api/generate-tests",
        headers={
            "Content-Type": "application/json"
        },
        json={
            "code": code,
            "problem_id": 1
        }
    )

    print("\nResponse Status:", response.status_code)
    
    try:
        # Parse the response - only once!
        test_cases = response.json()
        print("\nTest Cases:")
        print(test_cases)

        # Try to execute the generated test code
        print("\nTrying to execute the test code:")
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
        print(f"\nError: {e}")

if __name__ == "__main__":
    test_generate_tests()