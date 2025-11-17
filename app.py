from flask import Flask, request, jsonify
import joblib
import docx
import torch
import numpy as np
from model_mlp import SimpleNet
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
import fitz
import os
import uuid

app = Flask(__name__)
CORS(app)


# ==========  CARGAR MODELO SVM + TFIDF =========== #
svm_model = joblib.load("modelo_svm_calibrado.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# ==== CARGAR MODELO MLP ====
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ==== CARGAR MODELO MLP ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mlp_path = os.path.join(BASE_DIR, "mlp_model.pt")

print("Cargando modelo MLP desde:", mlp_path)
print("Archivos disponibles en BASE_DIR:", os.listdir(BASE_DIR))

mlp_model = SimpleNet(input_dim=384, hidden_dim=256, n_classes=3)

state_dict = torch.load(mlp_path, map_location="cpu")
mlp_model.load_state_dict(state_dict)
mlp_model.eval()

def extract_text_from_file(file):
    filename = file.filename.lower()
    ext = filename.split(".")[-1]
    tmp_path = f"/tmp/{uuid.uuid4()}.{ext}"  # NOMBRE ÚNICO

    file.save(tmp_path)

    try:
        if ext == "pdf":
            doc = fitz.open(tmp_path)
            text = "".join([page.get_text() for page in doc])
        elif ext == "docx":
            doc = docx.Document(tmp_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == "txt":
            text = open(tmp_path, "r", encoding="utf-8", errors="ignore").read()
        else:
            raise ValueError("Formato no soportado")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)  # LIMPIA SIEMPRE

    return text

#=============       SVM       =============#
@app.route("/predict/svm", methods=["POST"])
def predictSVM():
    if "file" not in request.files:
        return jsonify({"error": "No enviaste archivo"}), 400

    file = request.files["file"]

    try:
        text = extract_text_from_file(file)
        X = vectorizer.transform([text])
        prediction = int(svm_model.predict(X)[0])
        prob = svm_model.predict_proba(X)[0][1]     # probabilidad clase 1 (IA

        return jsonify({
            "predicted_label": prediction,
            "prob": prob
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
#=============       MLP       =============#
@app.route("/predict/mlp", methods=["POST"])
def predictMLP():
    if "file" not in request.files:
        return jsonify({"error": "Debes enviar un archivo"}), 400

    try:
        text = extract_text_from_file(request.files["file"])
        emb = embedder.encode([text], convert_to_numpy=True)

        X = torch.tensor(emb, dtype=torch.float32)
        logits = mlp_model(X)
        pred = torch.argmax(logits, dim=1).item()

        return jsonify({"prediction": int(pred)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/healthcheck")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
