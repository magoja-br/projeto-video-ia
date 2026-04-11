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

API_BASE = f"https://{LOCATION}-aiplatform.googleapis.com"

PREDICT_URL = (
    f"{API_BASE}/v1/projects/{PROJECT_ID}"
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
        print(f"[JOB {job_id}] Resposta completa: {json.dumps(resp_data, indent=2)[:800]}")

        operation_name = resp_data.get("name")
        if not operation_name:
            print(f"[JOB {job_id}] Sem operation name. Resposta: {resp.text[:500]}")
            jobs[job_id] = {"status": "error",
                            "error": "API não retornou nome da operação.",
                            "debug_response": resp.text[:500]}
            return

        print(f"[JOB {job_id}] Operation name: {operation_name}")

        # Salva debug info no job para poder consultar via /status
        jobs[job_id].update({
            "progress": 30,
            "message": "Gerando vídeo com IA...",
            "debug_operation_name": operation_name,
            "debug_initial_response": json.dumps(resp_data)[:500],
        })

        # ── 2. Polling da operação ────────────────────────────────────────
        # O Vertex AI Veo usa POST :fetchPredictOperation para polling.
        # NÃO usa GET no operation URL (isso retorna 404).
        #
        # Endpoint: POST .../models/{MODEL}:fetchPredictOperation
        # Body:     { "operationName": "<operation_name completo>" }

        fetch_url = (
            f"{API_BASE}/v1/projects/{PROJECT_ID}/locations/{LOCATION}"
            f"/publishers/google/models/{MODEL}:fetchPredictOperation"
        )

        jobs[job_id]["debug_fetch_url"] = fetch_url
        print(f"[JOB {job_id}] Fetch URL: {fetch_url}")

        max_tentativas = 180  # ~15 minutos (180 × 5s)

        for i in range(max_tentativas):
            time.sleep(5)

            # Renova token a cada ~10 tentativas (50s)
            if i % 10 == 9:
                token = get_token()
                print(f"[JOB {job_id}] Token renovado (tentativa {i+1})")

            try:
                poll_resp = req.post(
                    fetch_url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={"operationName": operation_name},
                    timeout=30,
                )
            except Exception as poll_err:
                print(f"[JOB {job_id}] Erro de conexão no polling (tentativa {i+1}): {poll_err}")
                jobs[job_id].update({"message": "Reconectando...", "debug_last_error": str(poll_err)})
                continue

            # Salva info de debug
            jobs[job_id]["debug_last_poll_status"] = poll_resp.status_code
            jobs[job_id]["debug_last_poll_response"] = poll_resp.text[:300]
            jobs[job_id]["debug_poll_attempt"] = i + 1

            if poll_resp.status_code != 200:
                print(f"[JOB {job_id}] Polling HTTP {poll_resp.status_code} (tentativa {i+1}): {poll_resp.text[:200]}")
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
                print(f"[JOB {job_id}] Sem predictions. Resposta: {json.dumps(poll_data, indent=2)[:500]}")
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
                gcs_url = gcs_uri
                if gcs_url.startswith("gs://"):
                    gcs_url = gcs_url.replace("gs://", "https://storage.googleapis.com/", 1)
                gcs_resp = req.get(
                    gcs_url,
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

            print(f"[JOB {job_id}] Formato desconhecido: {json.dumps(pred, indent=2)[:500]}")
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
        import traceback
        traceback.print_exc()
        jobs[job_id] = {"status": "error", "error": str(e)}


# ── Rotas ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "version": "3.0.0",
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "ia_model": MODEL,
    })


@app.route("/test-auth", methods=["GET"])
def test_auth():
    """
    Testa autenticação e conectividade com a API Vertex AI
    SEM gerar vídeo (não consome créditos).
    """
    resultado = {"version": "2.0.0"}

    # 1. Testar autenticação
    try:
        token = get_token()
        resultado["auth"] = "ok"
        resultado["token_preview"] = token[:20] + "..."
    except Exception as e:
        resultado["auth"] = "error"
        resultado["auth_error"] = str(e)
        return jsonify(resultado), 500

    # 2. Testar acesso ao endpoint (OPTIONS/GET simples, não POST)
    try:
        test_url = (
            f"{API_BASE}/v1/projects/{PROJECT_ID}"
            f"/locations/{LOCATION}/publishers/google/models/{MODEL}"
        )
        resp = req.get(
            test_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        resultado["api_access"] = f"HTTP {resp.status_code}"
        resultado["api_url_tested"] = test_url
        if resp.status_code != 200:
            resultado["api_response"] = resp.text[:300]
    except Exception as e:
        resultado["api_access"] = "error"
        resultado["api_error"] = str(e)

    # 3. Montar a URL de polling que seria usada (para debug)
    fake_op = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL}/operations/FAKE-ID"
    resultado["poll_url_format"] = f"{API_BASE}/v1/{fake_op}"
    resultado["predict_url"] = PREDICT_URL

    return jsonify(resultado)


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