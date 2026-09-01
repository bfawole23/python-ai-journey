from google import genai
import os

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents = "In exactly 3 bullet points, explain what Type 2 diabetes is, aimed at a patient with no medical background."
)

parsed = json.loads(response.text)
print(parsed["risk_level"])
print(parsed["condition"])
print(parsed["summary"])

print(response.text)