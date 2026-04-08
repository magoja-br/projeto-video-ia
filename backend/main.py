import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import vertexai

# Tenta importar de diferentes caminhos para evitar o ImportError
try:
    from vertexai.vision_models import ImageVideoModel
    print("LOG: Importado de vertexai.vision_models")
except ImportError:
    try:
        from vertexai.preview.vision_models import ImageVideoModel
        print("LOG: Importado de vertexai.preview.vision_models")
    except ImportError:
        ImageVideoModel = None
        print("LOG: ERRO - Nao foi possivel importar ImageVideoModel")

app = Flask(__name__)

# Configuração de CORS para aceitar qualquer origem
CORS(app, resources={r"/*": {"origins": "*"}})

# --- 1. CONFIGURAÇÃO DE PORTA ---
port = int(os.environ.get("PORT", 10000))

# --- 2. CONFIGURAÇÃO DE CREDENCIAIS (DIAGNÓSTICO MELHORADO) ---
render_secret_path = "/etc/secrets/google-credentials.json"
if os.path.exists(render_secret_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = render_secret_path
    print(f"LOG: Chave de segredo encontrada em {render_secret_path}")
else:
    print("LOG: Chave de segredo NAO encontrada no caminho do Render.")
    # Fallback local
    if os.path.exists("chave-google.json"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "chave-google.json"
    elif os.path.exists("../chave-google.json"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../chave-google.json"

# --- 3. INICIALIZAÇÃO VERTEX AI ---
PROJECT_ID = os.environ.get("PROJECT_ID", "gerador-de-imagens-ai") 
LOCATION = os.environ.get("LOCATION", "us-central1")

print(f"LOG: Iniciando Vertex AI no projeto {PROJECT_ID} em {LOCATION}")

model = None
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    if ImageVideoModel:
        # Tentamos carregar o modelo
        model = ImageVideoModel.from_pretrained("imagen-video")
        print("LOG: Modelo Imagen Video carregado com SUCESSO!")
    else:
        print("LOG: Modelo nao carregado porque a importacao falhou anteriormente.")
except Exception as e:
    print(f"LOG ERRO CRITICO AO CARREGAR MODELO: {str(e)}")
    model = None

# Pasta para salvar os vídeos
OUTPUT_FOLDER = os.path.join(os.getcwd(), "static")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/')
def home():
    status_ia = "CARREGADO" if model else "NAO CARREGADO"
    return jsonify({
        "status": "online", 
        "ia_model": status_ia,
        "project_id": PROJECT_ID,
        "location": LOCATION
    })

@app.route('/generate', methods=['POST'])
def generate():
    if not model:
        return jsonify({
            "error": "Modelo de video nao carregado no servidor.",
            "dica": "Verifique os logs do Render para ver o erro do Google Cloud."
        }), 500
        
    data = request.json
    prompt = data.get('prompt') if data else None
    
    if not prompt: 
        return jsonify({"error": "Digite um prompt!"}), 400
        
    try:
        print(f"LOG: Gerando video para o prompt: {prompt}")
        video = model.generate_video(prompt=prompt)
        
        filename = "clip_resultado.mp4"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        
        video.save(location=filepath)
        
        return jsonify({
            "success": True,
            "video_url": f"/static/{filename}"
        })
    except Exception as e:
        print(f"LOG ERRO NA GERACAO: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(OUTPUT_FOLDER, path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)