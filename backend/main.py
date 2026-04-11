import os
import time
import uuid
import base64
import threading
import tempfile
import json

import requests as req
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import google.auth
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import service_account

app = Flask(__name__)
CORS(app)

# ── Configuração ──────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("PROJECT_ID", "gerador-de-imagens-ai")
LOCATION   = os.environ.get("LOCATION", "us-central1")
MODEL      = "veo-3.1-fast-generate-001"

PREDICT_URL = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
    f"/locations/{LOCATION}/publishers/google/models/{MODEL}:predictLongRunning"
)
POLL_BASE = f"https://{LOCATION}-aiplatform.googleapis.com/v1/"

# Armazena jobs em memória
jobs: dict = {}


# ── Autenticação ──────────────────────────────────────────────────────────────
def get_token() -> str:
    """
    Retorna um Bearer Token válido.
    Suporta:
      1. GOOGLE_SERVICE_ACCOUNT_JSON  (conteúdo JSON como string)
      2. GOOGLE_APPLICATION_CREDENTIALS  (caminho para arquivo JSON)
      3. google.auth.default() como fallback
    """
    sa_json_str = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

    if sa_json_str:
        sa_info = json.loads(sa_json_str)
        creds = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    else:
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    creds.refresh(GoogleRequest())
    return creds.token


# ── Tarefa de geração (thread separada) ───────────────────────────────────────
def gerar_video(job_id: str, prompt: str, duration: int):
    try:
        jobs[job_id].update({"status": "processing", "progress": 10,
                             "message": "Autenticando..."})

        token = get_token()

        jobs[job_id].update({"progress": 20, "message": "Enviando requisição ao VEO..."})

        # ── 1. Dispara a geração ──────────────────────────────────────────
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "aspectRatio": "16:9",
                "sampleCount": 1,
                "durationSeconds": duration,
            },
        }

        resp = req.post(
            PREDICT_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )

        if resp.status_code != 200:
            jobs[job_id] = {
                "status": "error",
                "error": f"Erro ao iniciar geração (HTTP {resp.status_code}): {resp.text}",
            }
            return

        operation_name = resp.json().get("name")
        if not operation_name:
            jobs[job_id] = {"status": "error",
                            "error": "API não retornou nome da operação."}
            return

        jobs[job_id].update({"progress": 30, "message": "Gerando vídeo com IA..."})

        # ── 2. Polling da operação ────────────────────────────────────────
        poll_url = POLL_BASE + operation_name
        max_tentativas = 72  # ~6 minutos (72 × 5s)

        for i in range(max_tentativas):
            time.sleep(5)

            # Renova token a cada ~10 tentativas (50s)
            if i % 10 == 9:
                token = get_token()

            poll_resp = req.get(
                poll_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )

            if poll_resp.status_code != 200:
                continue

            poll_data = poll_resp.json()

            # Atualiza progresso
            pct = min(30 + int(i * 65 / max_tentativas), 95)
            if pct < 50:
                msg = "Preparando modelo..."
            elif pct < 70:
                msg = "Gerando frames..."
            elif pct < 88:
                msg = "Renderizando vídeo..."
            else:
                msg = "Finalizando..."
            jobs[job_id].update({"progress": pct, "message": msg})

            if not poll_data.get("done"):
                continue

            # ── 3. Operação concluída ─────────────────────────────────────
            if poll_data.get("error"):
                err = poll_data["error"]
                jobs[job_id] = {
                    "status": "error",
                    "error": err.get("message", "Erro desconhecido da API."),
                }
                return

            predictions = (
                poll_data.get("response", {}).get("predictions", [])
            )

            if not predictions:
                jobs[job_id] = {"status": "error",
                                "error": "API não retornou nenhum vídeo."}
                return

            pred = predictions[0]

            # Caso A: vídeo em base64
            video_b64 = pred.get("bytesBase64Encoded")
            if video_b64:
                video_bytes = base64.b64decode(video_b64)
                filepath = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
                with open(filepath, "wb") as f:
                    f.write(video_bytes)
                jobs[job_id] = {
                    "status": "done",
                    "progress": 100,
                    "message": "Concluído!",
                    "video_url": f"/video/{job_id}",
                }
                return

            # Caso B: URI do GCS
            gcs_uri = pred.get("gcsUri") or pred.get("videoUri")
            if gcs_uri:
                # Baixa o vídeo do GCS autenticado
                gcs_resp = req.get(
                    gcs_uri.replace("gs://", "https://storage.googleapis.com/"),
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=120,
                )
                if gcs_resp.status_code == 200:
                    filepath = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
                    with open(filepath, "wb") as f:
                        f.write(gcs_resp.content)
                    jobs[job_id] = {
                        "status": "done",
                        "progress": 100,
                        "message": "Concluído!",
                        "video_url": f"/video/{job_id}",
                    }
                    return

            jobs[job_id] = {                    
                "status": "error",
                "error": "Formato de resposta desconhecido. Dados: " + str(pred)[:300],
            }
            return

        # Timeout
        jobs[job_id] = {
            "status": "error",
            "error": "Tempo esgotado. A geração demorou mais de 6 minutos.",
        }

    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}


# ── Rotas ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "ia_model": MODEL,
    })


@app.route("/generate", methods=["POST"])
def generate():
    data     = request.json or {}
    prompt   = data.get("prompt", "").strip()
    duration = int(data.get("duration", 4))

    if not prompt:
        return jsonify({"error": "O campo 'prompt' é obrigatório."}), 400

    if duration not in (4, 5, 6, 8, 10):
        duration = 4

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": 0, "message": "Na fila..."}

    thread = threading.Thread(
        target=gerar_video, args=(job_id, prompt, duration), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job não encontrado."}), 404
    return jsonify(job)


@app.route("/video/<job_id>", methods=["GET"])
def serve_video(job_id):
    filepath = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
    if not os.path.exists(filepath):
        return jsonify({"error": "Vídeo não encontrado."}), 404
    return send_file(filepath, mimetype="video/mp4")


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)