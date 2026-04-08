import json
import requests
import google.auth
from google.auth.transport.requests import Request

# Autenticação
credentials, _ = google.auth.default()
credentials.refresh(Request())
token = credentials.token

# Endpoint do modelo VEO
url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/veo:predict"

# Corpo da requisição
payload = {
    "instances": [
        {
            "prompt": "Seu prompt aqui"
        }
    ]
}

# Chamada REST
response = requests.post(
    url,
    headers={"Authorization": f"Bearer {token}"},
    json=payload
)

print(response.json())