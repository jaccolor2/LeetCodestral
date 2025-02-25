import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

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
                "model": "mistral-moderation-latest"  # Updated to use latest moderation model
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

# Test the function with different content
import asyncio

async def main():
    test_contents = [
        "Hello, can you help me with this coding problem?",  # Safe content
        "I hate all people from that country, they shouldn't be allowed here",  # Hate speech
        "Here's my approach to solving the two sum problem",  # Safe content
        "I will find you and hurt you if you don't help me",  # Violence/threats
        "Here's my credit card number: 4111-1111-1111-1111",  # PII
        "I'm feeling really depressed and thinking of ending it",  # Self-harm
        "I know a way to hack into bank accounts",  # Dangerous/criminal
        "I'm stuck on this recursive function",  # Safe content
    ]

    for content in test_contents:
        print("\n" + "="*50)
        result = await check_moderation(content)
        print(f"Content: {content}")
        print(f"Is Safe: {result}")
        print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main()) 