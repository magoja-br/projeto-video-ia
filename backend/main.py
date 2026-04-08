import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import vertexai

# Tenta importar de diferentes caminhos para evitar o ImportError
try:
    from vertexai.vision_models import ImageVideoModel
except ImportError:
    try:
        from vertexai.preview.vision_models import ImageVideoModel
    except ImportError:
        ImageVideoModel = None

app = Flask(__name__)

# AJUSTE 1: Configuração de CORS mais forte para evitar "Erro ao conectar"
CORS(app, resources={r"/*": {"origins": "*"}})

# --- 1. CONFIGURAÇÃO DE PORTA ---
port = int(os.environ.get("PORT", 10000))

# --- 2. CREDENCIAIS ---
render_secret_path = "/etc/secrets/google-credentials.json"
if os.path.exists(render_secret_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = render_secret_path
else:
    # Se rodar local, tenta a pasta raiz ou a pasta backend
    if os.path.exists("chave-google.json"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "chave-google.json"
    elif os.path.exists("../chave-google.json"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../chave-google.json"

# --- 3. INICIALIZAÇÃO VERTEX AI ---
PROJECT_ID = os.environ.get("PROJECT_ID", "gerador-de-imagens-ai") 
LOCATION = os.environ.get("LOCATION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Inicializa o modelo
model = None
if ImageVideoModel:
    try:
        # Usando o modelo imagen-video (verifique se sua conta tem acesso a este modelo)
        model = ImageVideoModel.from_pretrained("imagen-video")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")

# Pasta para salvar os vídeos
OUTPUT_FOLDER = os.path.join(os.getcwd(), "static")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/')
def home():
    return jsonify({"status": "online", "message": "Servidor de Video IA pronto!"})

@app.route('/generate', methods=['POST'])
def generate():
    if not model:
        return jsonify({"error": "Modelo de video nao carregado. Verifique as credenciais e APIs."}), 500
        
    data = request.json
    if not data:
        return jsonify({"error": "JSON invalido"}), 400
        
    prompt = data.get('prompt')
    if not prompt: 
        return jsonify({"error": "Digite um prompt!"}), 400
        
    try:
        # AJUSTE 2: Geração do vídeo
        # Nota: O processamento pode demorar até 2 minutos no Vertex
        video = model.generate_video(prompt=prompt)
        
        filename = "clip_resultado.mp4"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        
        # Salva o vídeo no servidor
        video.save(location=filepath)
        
        # Retorna a URL (O Render vai servir via rota /static/)
        return jsonify({
            "success": True,
            "video_url": f"/static/{filename}"
        })
    except Exception as e:
        print(f"Erro interno: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(OUTPUT_FOLDER, path)

if __name__ == '__main__':
    # AJUSTE 3: 0.0.0.0 é obrigatório para o Render enxergar o Flask
    app.run(host='0.0.0.0', port=port)