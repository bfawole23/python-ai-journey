from google import genai
import os
import time
import json
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def call_with_retry(contents, max_retries=2):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=contents
            )
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    raise Exception("All retry attempts failed")

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return np.array(result.embeddings[0].values)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

knowledge_base = [
    "VPN connection issues: Restart the VPN client, confirm you're on a stable internet connection, and verify your credentials haven't expired.",
    "Password reset: Go to the self-service portal at portal.company.com/reset, enter your employee ID, and follow the email verification steps.",
    "Printer not responding: Check the printer is powered on and connected to the network. Restart the print spooler service on your machine.",
    "Slow computer performance: Restart the machine, check for pending updates, and close unused background applications."
]

def provide_fix(issue_summary):
    return f"Here's a suggested fix: {issue_summary}"

def escalate_to_technician(issue_description):
    return f"This issue has been logged and escalated to a technician: '{issue_description}'"

@app.route("/")
def home():
    return "IT Support AI Agent is running!"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    issue = data.get("question", "")

    prompt = f"""An employee reports: "{issue}" 
Decide the action:
- Use "provide_fix" ONLY for common, self-resolvable issues (VPN hiccups, printer issues, password resets, slow performance) with no major time pressure.
- Use "escalate_to_technician" if the issue involves: hardware failure, complete inability to work, urgent business deadlines/client-facing situations, or anything a quick fix likely won't solve in time.

Return ONLY valid JSON in this exact format, no other text:
{{
  "action": "provide_fix" or "escalate_to_technician",
  "reason": "why this action was chosen"
}}"""
    response = call_with_retry(prompt)
    decision = json.loads(response.text)

    if decision["action"] == "provide_fix":
        result = provide_fix(issue)
    else:
        result = escalate_to_technician(issue)

    return jsonify({"decision": decision, "result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

