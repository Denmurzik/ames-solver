import requests
import json

API_KEY = "sk-Hou2Ci2dLLEZblDfTGM30PatSWUq9WhKD1YrHZnK64Boa"
MODEL = "gemini-3-flash"
BASE_URLS_TO_TEST = [
    "http://127.0.0.1:8317/v1/models",
    "http://127.0.0.1:8317/v1beta/models",
    "http://127.0.0.1:8317/v1",
    "http://127.0.0.1:8317/v1beta"
]

payload = {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]}
headers = {"Content-Type": "application/json"}

for base in BASE_URLS_TO_TEST:
    url = f"{base}/{MODEL}:generateContent?key={API_KEY}"
    print(f"Testing {url}")
    try:
        r = requests.post(url, headers=headers, json=payload)
        print("Status:", r.status_code)
        if r.status_code == 200:
            print("Success! Response:", r.text[:100])
            break
        else:
            print("Response:", r.text[:100])
    except Exception as e:
        print("Exception:", e)
    print("-" * 40)
