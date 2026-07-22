# import anthropic
# import os

# # client= anthropic.Anthropic creates the connection object. My personal authentications to anthropic server
# client = anthropic.Anthropic(
#     api_key=os.environ.get("ANTHROPIC_API_KEY")
# )
# # client.messages.create(...) — using your authenticated connection to create a new message/conversation
# response = client.messages.create(
# # model="claude-sonnet-4-5" — telling Anthropic which AI model you want answering (there are several, varying in speed/cost/capability)
#     model="claude-sonnet-4-5",
# # max_tokens=200 — a limit on how long the response can be (tokens are roughly chunks of words) — keeps costs controlled, especially important while you're on limited free credit
#     max_tokens=200,
# # "role": "user" — marks this message as coming from you (the human), as opposed to "assistant" (Claude's own past replies, in a longer conversation)
# # "content": "..." — the actual text of your message/question
# # response — the variable storing whatever Claude sends back
#     messages=[
#         {"role": "user", "content": "Explain what a healthcare AI agent could do, in 2 sentences."}
#     ]
# )

# print(response.content[0].text)
from google import genai
import os

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Explain what a healthcare AI agent could do, in 2 sentences."
)

print(response.text)