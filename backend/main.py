import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import vertexai
# Importação para geração de VÍDEO (ajustada para Vertex AI)
from vertexai.vision_models import ImageVideoModel

app = Flask(__name__)
CORS(app)

# --- 1. CONFIGURAÇÃO DE PORTA (CORRIGE O ERRO DO RENDER) ---
# O Render passa a porta 10000 automaticamente por aqui
port = int(os.environ.get("PORT", 5000))

# --- 2. CONFIGURAÇÃO DE CREDENCIAIS ---
render_secret_path = "/etc/secrets/google-credentials.json"
if os.path.exists(render_secret_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = render_secret_path
else:
    # Se rodar local, procura o arquivo na sua pasta
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "chave-google.json"

# --- 3. INICIALIZAÇÃO VERTEX AI ---
# Pega o ID que você salvou no painel 'Environment' do Render
PROJECT_ID = os.environ.get("PROJECT_ID", "gerador-de-imagens-ai") 
LOCATION = os.environ.get("LOCATION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = ImageVideoModel.from_pretrained("imagen-video")

OUTPUT_FOLDER = "static"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/')
def home():
    return "Servidor de Vídeo Online!"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    if not prompt: 
        return jsonify({"error": "Digite um prompt!"}), 400
    try:
        # Gera o vídeo
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
    # Roda na porta correta para o Render não dar erro
    app.run(host='0.0.0.0', port=port)