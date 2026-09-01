
# import json

from google import genai
import os
import time

# client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# def call_with_retry(contents, max_retries=3):
#     for attempt in range(max_retries):
#         try:
#             response = client.models.generate_content(
#                 model="gemini-3.5-flash",
#                 contents="""Return ONLY valid JSON, no other text, in this exact format:
#                 {
#                 "condition": "name of condition"
#                 "risk_level": "low, medium, or high",
#                 "summary": "one sentence summary"
#                 }

#                 Analyze: A 65-year-old patient with high blood pressure and irregular exercise habits."""
                
#             )
#             return response
#         expect Exception as e:
#             print("Attempt {attempt + 1} failed: {e}")
#             if attempt < max_retries - 1:
#                 print("Waiting 15 seconds before retrying...")
#                 time.sleep(15)
#     raise Exception("All retry attempts failed")


# json.loads(response.text) 
# parsed["risk_level"]



# client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# response = client.models.generate_content(
#     model="gemini-3.5-flash",
#     contents="""Return ONLY valid JSON, no other text, in this exact format:
#     {
#     "condition": "name of condition"
#     "risk_level": "low, medium, or high",
#     "summary": "one sentence summary"
#     }

#     Analyze: A 65-year-old patient with high blood pressure and irregular exercise habits."""
# )

# json.loads(response.text)
# parsed["risk_level"]


import json

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def call_with_retry(contents, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=contents
            )
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Waiting 15 seconds before retrying...")
                time.sleep(15)
    raise Exception("All retry attempts failed")

prompt = """Return ONLY valid JSON, no other text, in this exact format:
{
  "condition": "name of condition",
  "risk_level": "low, medium, or high",
  "summary": "one sentence summary"
}

Analyze: A 65-year-old patient with high blood pressure and irregular exercise habits."""

response = call_with_retry(prompt)
print(response.text)
