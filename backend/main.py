import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import vertexai

# Tenta importar de diferentes caminhos para evitar o ImportError
try:
    from vertexai.generative_models import GenerativeModel
    # Para modelos de vídeo mais recentes (como o Veo ou Imagen Video)
    from vertexai.vision_models import ImageVideoModel
except ImportError:
    try:
        from vertexai.preview.vision_models import ImageVideoModel
    except ImportError:
        ImageVideoModel = None

app = Flask(__name__)
CORS(app)

# --- 1. CONFIGURAÇÃO DE PORTA (CORRIGE O ERRO DE PORTA) ---
port = int(os.environ.get("PORT", 5000))

# --- 2. CREDENCIAIS ---
render_secret_path = "/etc/secrets/google-credentials.json"
if os.path.exists(render_secret_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = render_secret_path
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "chave-google.json"

# --- 3. INICIALIZAÇÃO VERTEX AI ---
PROJECT_ID = os.environ.get("PROJECT_ID", "gerador-de-imagens-ai") 
LOCATION = os.environ.get("LOCATION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Inicializa o modelo com segurança
model = None
if ImageVideoModel:
    try:
        model = ImageVideoModel.from_pretrained("imagen-video")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")

OUTPUT_FOLDER = "static"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/')
def home():
    return "Servidor de Vídeo Online e Porta Configurada!"

@app.route('/generate', methods=['POST'])
def generate():
    if not model:
        return jsonify({"error": "Modelo de vídeo não disponível nesta região ou versão."}), 500
        
    data = request.json
    prompt = data.get('prompt')
    if not prompt: 
        return jsonify({"error": "Digite um prompt!"}), 400
    try:
        video = model.generate_video(prompt=prompt)
        filename = "clip_resultado.mp4"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        video.save(location=filepath)
        return jsonify({"video_url": f"/static/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(OUTPUT_FOLDER, path)

if __name__ == '__main__':
    # O host '0.0.0.0' e a porta variável são fundamentais no Render
    app.run(host='0.0.0.0', port=port)