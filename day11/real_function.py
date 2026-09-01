from google import genai
import os
import time
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

def check_bp_risk(systolic, diastolic):
    if systolic >= 140 or diastolic >= 90:
        return "High blood pressure risk - recommend follow-up."
    return "Blood pressure within normal range."

def schedule_followup(patient_name):
    return f"Follow-up appointment scheduled for {patient_name}."

prompt = """A patient has systolic blood pressure of 150 and diastolic of 95.

Return ONLY valid JSON in this exact format, no other text:
{
  "action": "check_bp_risk",
  "reason": "why this action was chosen"
}"""

response = call_with_retry(prompt)
decision = json.loads(response.text)
print(decision)

if decision["action"] == "check_bp_risk":
    result = check_bp_risk(150, 95)
    print(result)
elif decision["action"] == "schedule_followup":
    result = schedule_followup("Amaka")
    print(result)