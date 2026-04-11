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
def gerar_video(job_id: str, prompt: str, duration: int, aspect_ratio: str):
    try:
        jobs[job_id].update({"status": "processing", "progress": 10,
                             "message": "Autenticando..."})

        token = get_token()

        jobs[job_id].update({"progress": 20, "message": "Enviando requisição ao VEO..."})

        # ── 1. Dispara a geração ──────────────────────────────────────────
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "aspectRatio": aspect_ratio,
                "sampleCount": 1,
                "durationSeconds": duration,
            },
        }

        print(f"[JOB {job_id}] Prompt: {prompt[:100]}")
        print(f"[JOB {job_id}] Params: duration={duration}, ratio={aspect_ratio}, model={MODEL}")
        print(f"[JOB {job_id}] URL: {PREDICT_URL}")

        resp = req.post(
            PREDICT_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )

        print(f"[JOB {job_id}] Resposta inicial: HTTP {resp.status_code}")

        if resp.status_code != 200:
            print(f"[JOB {job_id}] ERRO: {resp.text[:500]}")
            jobs[job_id] = {
                "status": "error",
                "error": f"Erro ao iniciar geração (HTTP {resp.status_code}): {resp.text[:300]}",
            }
            return

        resp_data = resp.json()
        print(f"[JOB {job_id}] Resposta completa: {str(resp_data)[:800]}")

        operation_name = resp_data.get("name")
        if not operation_name:
            print(f"[JOB {job_id}] Sem operation name. Resposta: {resp.text[:500]}")
            jobs[job_id] = {"status": "error",
                            "error": "API não retornou nome da operação."}
            return

        print(f"[JOB {job_id}] Operation name: {operation_name}")
        jobs[job_id].update({"progress": 30, "message": "Gerando vídeo com IA..."})

        # ── 2. Polling da operação ────────────────────────────────────────
        api_base = f"https://{LOCATION}-aiplatform.googleapis.com"

        # A API retorna operation_name no formato:
        # projects/.../locations/.../publishers/google/models/.../operations/{id}
        # Mas o endpoint de polling (fetchOperation) precisa de:
        # projects/.../locations/.../operations/{id}

        # Extrai o operation_id do final do operation_name
        op_id = operation_name.split("/operations/")[-1] if "/operations/" in operation_name else ""
        simplified_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/operations/{op_id}"

        print(f"[JOB {job_id}] Operation ID: {op_id}")
        print(f"[JOB {job_id}] Simplified name: {simplified_name}")

        # Tenta múltiplos formatos de URL
        poll_urls = [
            f"{api_base}/v1/{simplified_name}",
            f"{api_base}/v1beta1/{simplified_name}",
            f"{api_base}/v1/{operation_name}",
            f"{api_base}/v1beta1/{operation_name}",
        ]

        # Se operation_name já contiver a URL completa, usar diretamente
        if operation_name.startswith("http"):
            poll_urls = [operation_name]

        # Detecta qual URL funciona na primeira tentativa
        poll_url = None
        max_tentativas = 180  # ~15 minutos (180 × 5s)

        print(f"[JOB {job_id}] URLs de polling para testar: {poll_urls}")

        for i in range(max_tentativas):
            time.sleep(5)

            # Renova token a cada ~10 tentativas (50s)
            if i % 10 == 9:
                token = get_token()
                print(f"[JOB {job_id}] Token renovado (tentativa {i+1})")

            # Tenta a URL de polling (ou descobre qual funciona)
            poll_resp = None
            urls_to_try = [poll_url] if poll_url else poll_urls

            for url in urls_to_try:
                try:
                    resp_test = req.get(
                        url,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=30,
                    )
                    if resp_test.status_code == 200:
                        poll_resp = resp_test
                        if poll_url != url:
                            poll_url = url
                            print(f"[JOB {job_id}] ✅ URL de polling encontrada: {url}")
                        break
                    else:
                        if i == 0:
                            print(f"[JOB {job_id}] URL {url} retornou HTTP {resp_test.status_code}")
                except Exception as poll_err:
                    if i == 0:
                        print(f"[JOB {job_id}] URL {url} erro: {poll_err}")

            if not poll_resp or poll_resp.status_code != 200:
                if i % 12 == 0:
                    print(f"[JOB {job_id}] Polling sem resposta válida (tentativa {i+1})")
                continue

            poll_data = poll_resp.json()

            # Atualiza progresso
            pct = min(30 + int(i * 65 / max_tentativas), 95)
            if pct < 50:
                msg = "Preparando modelo..."
            elif pct < 70:
                msg = "Gerando frames..."
            elif pct < 85:
                msg = "Renderizando vídeo..."
            elif pct < 93:
                msg = "Finalizando..."
            else:
                msg = "Quase pronto..."
            jobs[job_id].update({"progress": pct, "message": msg})

            if not poll_data.get("done"):
                # Log a cada 12 tentativas (~1 min)
                if i % 12 == 0:
                    print(f"[JOB {job_id}] Ainda processando... (tentativa {i+1}, {pct}%)")
                continue

            # ── 3. Operação concluída ─────────────────────────────────────
            print(f"[JOB {job_id}] Operação concluída! (tentativa {i+1})")

            if poll_data.get("error"):
                err = poll_data["error"]
                err_msg = err.get("message", "Erro desconhecido da API.")
                print(f"[JOB {job_id}] ERRO da API: {err_msg}")
                jobs[job_id] = {
                    "status": "error",
                    "error": err_msg,
                }
                return

            predictions = (
                poll_data.get("response", {}).get("predictions", [])
            )

            if not predictions:
                print(f"[JOB {job_id}] Sem predictions. Resposta: {str(poll_data)[:500]}")
                jobs[job_id] = {"status": "error",
                                "error": "API não retornou nenhum vídeo."}
                return

            pred = predictions[0]
            print(f"[JOB {job_id}] Prediction keys: {list(pred.keys())}")

            # Caso A: vídeo em base64
            video_b64 = pred.get("bytesBase64Encoded")
            if video_b64:
                video_bytes = base64.b64decode(video_b64)
                filepath = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
                with open(filepath, "wb") as f:
                    f.write(video_bytes)
                print(f"[JOB {job_id}] ✅ Vídeo salvo (base64): {len(video_bytes)} bytes")
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
                print(f"[JOB {job_id}] Baixando do GCS: {gcs_uri}")
                gcs_resp = req.get(
                    gcs_uri.replace("gs://", "https://storage.googleapis.com/"),
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=120,
                )
                if gcs_resp.status_code == 200:
                    filepath = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
                    with open(filepath, "wb") as f:
                        f.write(gcs_resp.content)
                    print(f"[JOB {job_id}] ✅ Vídeo salvo (GCS): {len(gcs_resp.content)} bytes")
                    jobs[job_id] = {
                        "status": "done",
                        "progress": 100,
                        "message": "Concluído!",
                        "video_url": f"/video/{job_id}",
                    }
                    return
                else:
                    print(f"[JOB {job_id}] Erro ao baixar GCS: HTTP {gcs_resp.status_code}")

            print(f"[JOB {job_id}] Formato desconhecido: {str(pred)[:500]}")
            jobs[job_id] = {                    
                "status": "error",
                "error": "Formato de resposta desconhecido. Dados: " + str(pred)[:300],
            }
            return

        # Timeout
        print(f"[JOB {job_id}] ⏰ TIMEOUT após {max_tentativas * 5}s")
        jobs[job_id] = {
            "status": "error",
            "error": "Tempo esgotado. A geração demorou mais de 15 minutos. Tente novamente com um prompt mais simples.",
        }

    except Exception as e:
        print(f"[JOB {job_id}] 💥 EXCEPTION: {e}")
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
    data         = request.json or {}
    prompt       = data.get("prompt", "").strip()
    duration     = int(data.get("duration", 4))
    aspect_ratio = data.get("aspect_ratio", "16:9")

    if not prompt:
        return jsonify({"error": "O campo 'prompt' é obrigatório."}), 400

    if duration not in (4, 5, 6, 8, 10):
        duration = 4

    if aspect_ratio not in ("16:9", "9:16"):
        aspect_ratio = "16:9"

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": 0, "message": "Na fila..."}

    thread = threading.Thread(
        target=gerar_video, args=(job_id, prompt, duration, aspect_ratio), daemon=True
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