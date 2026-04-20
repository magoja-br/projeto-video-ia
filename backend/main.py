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
MODEL_FAST = "veo-3.1-fast-generate-001"
MODEL_BAL  = "veo-3.1-generate-001"
# O modelo padrão continua sendo o fast
MODEL      = MODEL_FAST 
APP_PASSWORD = os.environ.get("APP_PASSWORD", "mudar-senha-123") # Senha padrão se não definida

API_BASE = f"https://{LOCATION}-aiplatform.googleapis.com"

PREDICT_URL = (
    f"{API_BASE}/v1/projects/{PROJECT_ID}"
    f"/locations/{LOCATION}/publishers/google/models/{MODEL}:predictLongRunning"
)

FETCH_URL = (
    f"{API_BASE}/v1/projects/{PROJECT_ID}"
    f"/locations/{LOCATION}/publishers/google/models/{MODEL}:fetchPredictOperation"
)

# Armazena jobs em memória
jobs: dict = {}


# ── Autenticação da App ───────────────────────────────────────────────────────
def check_auth():
    """Verifica se a senha enviada no header X-Password está correta."""
    if not APP_PASSWORD:
        return True # Sem senha configurada
    
    provided_password = request.headers.get("X-Password")
    return provided_password == APP_PASSWORD


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


# ── Otimização de Prompt ──────────────────────────────────────────────────────
def otimizar_prompt(original_prompt: str) -> str:
    """
    Usa o Gemini 1.5 Flash para traduzir e expandir o prompt para inglês,
    tornando-o mais descritivo e cinematográfico.
    """
    try:
        token = get_token()
        # Usamos o Gemini 1.5 Flash para ser rápido e eficiente
        url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/gemini-1.5-flash-002:generateContent"
        
        system_instruction = (
            "You are an expert prompt engineer for AI video generation (Google Veo). "
            "Your task is to take a user prompt (likely in Portuguese) and rewrite it as an optimized English video generation prompt. "
            "Follow these rules strictly:\n"
            "1. Translate to English.\n"
            "2. Expand into a rich, cinematic, and highly detailed prompt.\n"
            "3. Focus on: lighting quality (chiaroscuro, dramatic, soft), camera movements (slow crane, dolly zoom), "
            "film aesthetics (epic, wide-angle, anamorphic lens), color palette, and historical/period details.\n"
            "4. CRITICAL — Sensitive/Religious content: If the prompt involves religious scenes, themes of suffering, or sacred figures, "
            "you MUST reframe it using HIGHLY ABSTRACT and ARTISTIC language. "
            "Focus on: 'a central symbolic figure', 'dramatic lighting', 'chiaroscuro', 'heavenly rays', 'solemn atmosphere'. "
            "Avoid literal descriptions like 'wooden cross' or 'crucified'. Instead use 'a vertical beam', 'a silhouette against a stormy sky', "
            "or 'an epic masterwork painting come to life'. "
            "Describe the scene as a museum-quality historical exploration of light and shadow, inspired by Rembrandt, Velázquez, or Caravaggio. "
            "Focus on the 'ethereal glow' and 'dramatic clouds' rather than the specific religious context. "
            "Emphasize it as a 'historical epic fine-art recreation'.\n"
            "5. Return ONLY the final English prompt text, no explanations, no quotation marks around it."
        )
        
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": original_prompt}]
            }],
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 400,
            }
        }
        
        resp = req.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30
        )
        
        if resp.status_code == 200:
            data = resp.json()
            try:
                optimized = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if optimized:
                    return optimized
            except (KeyError, IndexError):
                pass
        
        print(f"Aviso: Falha ao otimizar prompt (HTTP {resp.status_code}). Usando original.")
    except Exception as e:
        print(f"Erro na otimização de prompt: {e}")
    
    return original_prompt


def gerar_video(job_id: str, prompt: str, duration: int, aspect_ratio: str, model_name: str):
    try:
        # Define os URLs dinamicamente baseado no modelo
        predict_url = (
            f"{API_BASE}/v1/projects/{PROJECT_ID}"
            f"/locations/{LOCATION}/publishers/google/models/{model_name}:predictLongRunning"
        )
        fetch_url = (
            f"{API_BASE}/v1/projects/{PROJECT_ID}"
            f"/locations/{LOCATION}/publishers/google/models/{model_name}:fetchPredictOperation"
        )

        jobs[job_id].update({"status": "processing", "progress": 5,
                             "message": "Otimizando prompt..."})

        prompt_en = otimizar_prompt(prompt)
        print(f"[JOB {job_id}] Original: {prompt}")
        print(f"[JOB {job_id}] Otimizado: {prompt_en}")

        jobs[job_id].update({"progress": 15, "message": "Autenticando...", "prompt_otimizado": prompt_en})

        token = get_token()

        # ── Preparação do Payload ──
        instance = {"prompt": prompt_en}
        
        # Se houver imagem de referência (base64)
        image_b64 = jobs[job_id].get("reference_image")
        if image_b64:
            # O Veo espera a imagem no campo 'image' dentro de 'instances'
            # Pode ser uma URL do GCS ou bytes em base64 (dependendo da versão exata)
            # Para o Veo 3.1, costuma ser via 'image' object
            instance["image"] = {
                "bytesBase64Encoded": image_b64
            }
            msg_envio = f"Enviando ao Veo (com imagem)... prompt: '{prompt_en[:40]}...'"
        else:
            msg_envio = f"Enviando ao Veo (apenas texto)... prompt: '{prompt_en[:40]}...'"

        jobs[job_id].update({"progress": 25, "message": msg_envio})

        # ── 1. Dispara a geração ──────────────────────────────────────────
        payload = {
            "instances": [instance],
            "parameters": {
                "aspectRatio": aspect_ratio,
                "sampleCount": 1,
                "durationSeconds": duration,
            },
        }

        print(f"[JOB {job_id}] Prompt: {prompt[:100]}")
        print(f"[JOB {job_id}] Params: duration={duration}, ratio={aspect_ratio}, model={model_name}")

        resp = req.post(
            predict_url,
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
        operation_name = resp_data.get("name")
        if not operation_name:
            print(f"[JOB {job_id}] Sem operation name. Resposta: {resp.text[:500]}")
            jobs[job_id] = {"status": "error",
                            "error": "API não retornou nome da operação."}
            return

        print(f"[JOB {job_id}] Operation name: {operation_name}")
        jobs[job_id].update({"progress": 30, "message": "Gerando vídeo com IA..."})

        # ── 2. Polling da operação ────────────────────────────────────────
        # O Vertex AI Veo usa POST :fetchPredictOperation para polling.
        # NÃO usa GET no operation URL (isso retorna 404).
        #
        # Endpoint: POST .../models/{MODEL}:fetchPredictOperation
        # Body:     { "operationName": "<operation_name completo>" }

        print(f"[JOB {job_id}] Fetch URL: {FETCH_URL}")

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
                jobs[job_id].update({"message": "Reconectando..."})
                continue

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
                if i % 12 == 0:
                    print(f"[JOB {job_id}] Ainda processando... (tentativa {i+1}, {pct}%)")
                continue

            # ── 3. Operação concluída ─────────────────────────────────────
            print(f"[JOB {job_id}] Operação concluída! (tentativa {i+1})")

            if poll_data.get("error"):
                err = poll_data["error"]
                err_msg = err.get("message", "Erro desconhecido da API.")
                print(f"[JOB {job_id}] ERRO da API: {err_msg}")
                jobs[job_id] = {"status": "error", "error": err_msg}
                return

            # Tenta extrair predictions de múltiplos formatos possíveis
            predictions = (
                poll_data.get("response", {}).get("predictions", [])
                or poll_data.get("predictions", [])
                or poll_data.get("result", {}).get("predictions", [])
            )

            if not predictions:
                predictions = (
                    poll_data.get("response", {}).get("videos", [])
                    or poll_data.get("videos", [])
                    or poll_data.get("result", {}).get("videos", [])
                )

            if not predictions:
                full_resp_str = json.dumps(poll_data)
                print(f"[JOB {job_id}] ⚠️ Sem resultados. Dados completos: {full_resp_str}")
                
                # Verifica se houve bloqueio por segurança
                safety_reason = ""
                safety_keywords = ["safetyAttributes", "safety_attributes", "blocked", "SAFETY", "PROHIBITED"]
                if any(kw in full_resp_str for kw in safety_keywords):
                    safety_reason = " (Filtro de segurança do Google ativado)"
                
                prompt_resumo = jobs[job_id].get("prompt_otimizado", "")[:120]
                jobs[job_id] = {
                    "status": "error",
                    "error": (
                        f"A IA bloqueou a geração do vídeo{safety_reason}. "
                        f"Prompt enviado: '{prompt_resumo}...'. "
                        f"Tente descrever a cena de forma diferente."
                    ),
                }
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

            print(f"[JOB {job_id}] Formato desconhecido: {str(pred)[:500]}")
            jobs[job_id] = {
                "status": "error",
                "error": "Formato de resposta desconhecido.",
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
    # Health check aberto para verificar se o serviço está vivo
    return jsonify({
        "status": "online",
        "version": "3.2.1",
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "ia_model": MODEL,
        "protected": bool(APP_PASSWORD)
    })


@app.route("/generate", methods=["POST"])
def generate():
    if not check_auth():
        return jsonify({"error": "Senha incorreta ou não fornecida."}), 401
    
    data         = request.json or {}
    prompt       = data.get("prompt", "").strip()
    duration     = int(data.get("duration", 4))
    aspect_ratio = data.get("aspect_ratio", "16:9")
    model_name   = data.get("model", MODEL_FAST)
    image_b64    = data.get("image") # Imagem em base64 (opcional)

    if model_name not in (MODEL_FAST, MODEL_BAL):
        model_name = MODEL_FAST

    if not prompt:
        return jsonify({"error": "O campo 'prompt' é obrigatório."}), 400

    if duration not in (4, 5, 6, 8, 10):
        duration = 4

    if aspect_ratio not in ("16:9", "9:16"):
        aspect_ratio = "16:9"

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued", 
        "progress": 0, 
        "message": "Na fila...",
        "reference_image": image_b64 # Salva a imagem para o worker usar
    }

    thread = threading.Thread(
        target=gerar_video, args=(job_id, prompt, duration, aspect_ratio, model_name), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    if not check_auth():
        return jsonify({"error": "Senha incorreta."}), 401
        
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job não encontrado."}), 404
    return jsonify(job)


@app.route("/video/<job_id>", methods=["GET"])
def serve_video(job_id):
    # Nota: Aqui a senha pode ser passada via query param (?password=...) 
    # se o <video> tag não suportar custom headers facilmente.
    # Mas para uso pessoal simples, vamos tentar via header ou query.
    pass_header = request.headers.get("X-Password")
    pass_query = request.args.get("password")
    
    if APP_PASSWORD and pass_header != APP_PASSWORD and pass_query != APP_PASSWORD:
        return jsonify({"error": "Acesso negado."}), 401
        
    filepath = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
    if not os.path.exists(filepath):
        return jsonify({"error": "Vídeo não encontrado."}), 404
    return send_file(filepath, mimetype="video/mp4")


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)