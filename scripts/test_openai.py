# FILE: scripts/test_openai.py
import sys
sys.path.append('..')

import os
from openai import OpenAI
from dotenv import load_dotenv  # <-- ADD THIS LINE

# Load the .env file explicitly
load_dotenv()  # <-- ADD THIS LINE

# Try to get the API key from environment variables
api_key = os.environ.get("OPENAI_API_KEY")

# Check if we found it
if not api_key:
    print("❌ ERROR: Could not find OPENAI_API_KEY")
    print("Make sure you created a .env file in the project root")
    print("with the line: OPENAI_API_KEY=sk-your-key")
    print(f"Current directory: {os.getcwd()}")  # <-- ADD THIS to debug
    exit()

print(f"✓ Found API key: {api_key[:10]}...")  # Show first 10 chars only

# Try to call OpenAI
print("\nTesting OpenAI API...")
try:
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say 'Hello, I am working!' in exactly 5 words."}
        ],
        max_tokens=50
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print("\n✓✓✓ API key works! Ready to build the LLM agent!")
    
except Exception as e:
    print(f"❌ Error calling OpenAI: {e}")
    print("Check that your API key is correct")