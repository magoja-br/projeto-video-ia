import os
import time
import uuid
import base64
import threading
import tempfile

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Configuração ──────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("PROJECT_ID", "gerador-de-imagens-ai")
LOCATION   = os.environ.get("LOCATION", "us-central1")
MODEL      = "veo-2.0-generate-001"

# Armazenamento em memória dos jobs (reinicia quando o servidor reinicia)
jobs: dict = {}

# ── Autenticação ──────────────────────────────────────────────────────────────
def get_client():
    """
    Retorna um cliente Google GenAI autenticado.
    Suporta duas formas de credencial:
      1. Variável de ambiente GOOGLE_SERVICE_ACCOUNT_JSON  (JSON completo como string)
      2. google.auth.default()  (Application Default Credentials)
    """
    import json
    from google import genai

    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        # Salva temporariamente o JSON em disco para que a lib consiga ler
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write(sa_json)
        tmp.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )
    return client

# ── Tarefa de geração (thread separada) ───────────────────────────────────────
def gerar_video(job_id: str, prompt: str, duration: int):
    try:
        from google.genai import types

        jobs[job_id].update({"status": "processing", "progress": 10,
                             "message": "Conectando à API..."})

        client = get_client()

        jobs[job_id].update({"progress": 20, "message": "Enviando requisição ao VEO..."})

        # Dispara a geração (Long-Running Operation)
        operation = client.models.generate_videos(
            model=MODEL,
            prompt=prompt,
            config=types.GenerateVideoConfig(
                aspect_ratio="16:9",
                number_of_videos=1,
                duration_seconds=duration,
            ),
        )

        jobs[job_id].update({"progress": 30, "message": "Gerando vídeo com IA..."})

        # Polling até concluir
        tentativas = 0
        max_tentativas = 72  # ~6 minutos com sleep de 5s

        while not operation.done and tentativas < max_tentativas:
            time.sleep(5)
            tentativas += 1
            operation = client.operations.get(operation.name)

            progresso = min(30 + int(tentativas * 65 / max_tentativas), 95)

            if progresso < 50:
                msg = "Preparando modelo..."
            elif progresso < 70:
                msg = "Gerando frames..."
            elif progresso < 88:
                msg = "Renderizando vídeo..."
            else:
                msg = "Finalizando..."

            jobs[job_id].update({"progress": progresso, "message": msg})

        # Verificação de timeout
        if not operation.done:
            jobs[job_id] = {"status": "error",
                            "error": "Tempo esgotado. A geração demorou demais."}
            return

        # Verificação de erro vindo da API
        if operation.error and operation.error.message:
            jobs[job_id] = {"status": "error", "error": operation.error.message}
            return

        # Extrai o vídeo gerado
        videos = operation.result.generated_videos if operation.result else []
        if not videos:
            jobs[job_id] = {"status": "error", "error": "Nenhum vídeo retornado pela API."}
            return

        video = videos[0]

        # Caso 1: URI de GCS
        if hasattr(video, "video") and video.video and video.video.uri:
            # Baixa o vídeo do GCS usando o cliente autenticado
            gcs_uri = video.video.uri  # gs://bucket/path.mp4
            video_bytes = client.files.download(file=video.video)
            filepath = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
            with open(filepath, "wb") as f:
                f.write(video_bytes)
            jobs[job_id] = {"status": "done", "progress": 100,
                            "message": "Concluído!", "video_url": f"/video/{job_id}"}
            return

        # Caso 2: Base64 direto
        if hasattr(video, "video") and video.video and hasattr(video.video, "video_bytes"):
            video_bytes = video.video.video_bytes
            filepath = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
            with open(filepath, "wb") as f:
                f.write(video_bytes)
            jobs[job_id] = {"status": "done", "progress": 100,
                            "message": "Concluído!", "video_url": f"/video/{job_id}"}
            return

        jobs[job_id] = {"status": "error",
                        "error": "Formato de resposta desconhecido da API."}

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
    data   = request.json or {}
    prompt = data.get("prompt", "").strip()
    duration = int(data.get("duration", 4))

    if not prompt:
        return jsonify({"error": "O campo 'prompt' é obrigatório."}), 400

    if duration not in (4, 5, 6, 8, 10):
        duration = 4

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": 0,
                    "message": "Na fila..."}

    thread = threading.Thread(target=gerar_video,
                              args=(job_id, prompt, duration),
                              daemon=True)
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