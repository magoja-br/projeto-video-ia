import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

app = Flask(__name__)
CORS(app)

# Configurações Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "chave-google.json"
PROJECT_ID = "seu-projeto-criador-de-videos" 
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = ImageGenerationModel.from_pretrained("imagegeneration@006")

OUTPUT_FOLDER = "static"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    duration = data.get('duration')
    if not prompt: return jsonify({"error": "Digite um prompt!"}), 400
    try:
        full_prompt = f"{prompt}. Duration: {duration} seconds, cinematic high quality, 4k."
        response = model.generate_images(prompt=full_prompt, number_of_images=1)
        filename = "clip_resultado.mp4"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        response[0].save(location=filepath, include_generation_parameters=False)
        return jsonify({"video_url": f"/static/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(OUTPUT_FOLDER, path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)