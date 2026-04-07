import os
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url=os.getenv("API_BASE_URL").strip(),
    api_key=os.getenv("HF_TOKEN").strip(),
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "hospital-triage-env"
    }
)

MODEL_NAME = os.getenv("MODEL_NAME")

# Simple test prompt
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ],
    temperature=0
)

print(response.choices[0].message.content)