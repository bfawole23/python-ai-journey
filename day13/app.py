from flask import Flask, request, jsonify
from google import genai
import os
import time
import json


# User
#   ↓
# Frontend
#   ↓
# Flask API
#   ↓
# Gemini API
#   ↓
# Flask API
#   ↓
# Frontend
#   ↓
# User


# Flask is the web server/backend part of your Python application.

# In your healthcare AI project, Flask acts as the middleman between your frontend and Gemini.
# Your frontend sends that question to Flask:
# POST /ask

app = Flask(__name__)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def call_with_retry(contents, max_retries=3):
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
                time.sleep(15)
    raise Exception("All retry attempts failed")

# route() is a Flask function that connects a URL to a Python function.
# This creates a URL endpoint called:
# The / is your home endpoint.
@app.route("/")
def home():
    return "Healthcare AI Agent is running!"
# is telling Flask what should happen when someone sends a request to /ask.
# "The /ask endpoint should accept POST requests."
@app.route("/ask", methods=["POST"])
# It tells Flask:

# "The /ask endpoint should accept POST requests."
# Flask receives it:
def ask():
    data = request.get_json()
    question = data.get("question", "")
    # Then Flask sends the question to Gemini:
    response = call_with_retry(question)
    return jsonify({"answer": response.text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

    # @app.route("/ask", methods=["POST"])
    #    │             │
    #    │             └── What type of request?
    #    │
    #    └──────────────── Where?