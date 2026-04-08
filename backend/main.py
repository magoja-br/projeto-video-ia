import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import vertexai
# Importação robusta para suportar diferentes versões da SDK
try:
    from vertexai.vision_models import ImageVideoModel
except ImportError:
    from vertexai.preview.vision_models import ImageVideoModel

app = Flask(__name__)
CORS(app)

# --- 1. CONFIGURAÇÃO DE SEGURANÇA E AUTENTICAÇÃO ---
# O Render armazena o JSON em /etc/secrets/ se você usou o "Secret Files"
render_secret_path = "/etc/secrets/google-credentials.json"
local_key_path = "chave-google.json"

if os.path.exists(render_secret_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = render_secret_path
    print("Log: Usando credenciais do Render (/etc/secrets).")
elif os.path.exists(local_key_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_key_path
    print(f"Log: Usando credenciais locais ({local_key_path}).")

# --- 2. INICIALIZAÇÃO DO VERTEX AI ---
# Lê as variáveis que você configurou no painel 'Environment' do Render
PROJECT_ID = os.environ.get("PROJECT_ID", "gerador-de-imagens-ai") 
LOCATION = os.environ.get("LOCATION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Carrega o modelo de geração de vídeo (Veo/Imagen Video)
try:
    model = ImageVideoModel.from_pretrained("imagen-video")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    model = None

# Configuração da pasta de saída
OUTPUT_FOLDER = "static"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# --- 3. ROTAS DA API ---

@app.route('/')
def index():
    return "Servidor de Geração de Vídeo IA está Online!"

@app.route('/generate', methods=['POST'])
def generate():
    if not model:
        return jsonify({"error": "Modelo não carregado no servidor."}), 500

    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "Por favor, digite um prompt!"}), 400
    
    try:
        # Geração do vídeo
        # Nota: O Vertex AI pode levar de 30 a 60 segundos para gerar.
        video = model.generate_video(
            prompt=prompt,
            # Você pode adicionar parâmetros como number_of_videos=1 se a SDK permitir
        )
        
        filename = "video_gerado.mp4"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        
        # Salva o vídeo no disco do servidor
        video.save(location=filepath)
        
        # Retorna a URL para o seu front-end
        return jsonify({
            "success": True,
            "video_url": f"/static/{filename}"
        })
    
    except Exception as e:
        print(f"Erro na geração: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(OUTPUT_FOLDER, path)

# --- 4. EXECUÇÃO ---
if __name__ == '__main__':
    # O Render exige que o app rode na porta definida pela variável de ambiente PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)