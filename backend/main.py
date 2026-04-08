import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import vertexai
from vertexai.preview.generative_models import GenerativeModel

app = Flask(__name__)
CORS(app)

PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)

try:
    model = GenerativeModel("veo-1.5")
    print("LOG: Modelo VEO carregado com sucesso!")
except Exception as e:
    print("ERRO ao carregar modelo:", e)
    model = None

OUTPUT_FOLDER = "static"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return jsonify({"status": "online", "model": "veo-1.5", "loaded": model is not None})

@app.route("/generate", methods=["POST"])
def generate():
    if not model:
        return jsonify({"error": "Modelo nao carregado no servidor"}), 500

    data = request.json
    prompt = data.get("prompt")
    duration = int(data.get("duration", 4))

    try:
        print("Gerando vídeo:", prompt)

        result = model.generate_video(
            prompt=prompt,
            video_duration=duration
        )

        filename = "resultado.mp4"
        filepath = os.path.join(OUTPUT_FOLDER, filename)

        with open(filepath, "wb") as f:
            f.write(result)

        return jsonify({"video_url": f"/static/{filename}"})

    except Exception as e:
        print("ERRO:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(OUTPUT_FOLDER, path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)