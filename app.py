from flask import Flask, request, jsonify
import joblib
import torch
from models.mlp.model_mlp import SimpleNet
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
#from text_extractor import extract_text_from_file
from dotenv import load_dotenv
load_dotenv()
import os

# para roberta
from safetensors.torch import load_file
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
import os

origins = os.getenv("CROSS_ORIGINS", "").split(",")
app = Flask(__name__)
CORS(
    app,
    origins=origins
)

# ==========  CARGAR MODELO SVM + TFIDF =========== #
svm_model = joblib.load("models/svm/modelo_svm_calibrado.joblib")
vectorizer = joblib.load("models/svm/tfidf_vectorizer.joblib")

#=============       ENDPOINT SVM       =============#
@app.route("/predict/svm", methods=["POST"])
def predictSVM():
    #if "file" not in request.files:
    #    return jsonify({"error": "No enviaste archivo"}), 400
    # file = request.files["file"]

    data = request.get_json(silent=True)
    if not data:
        return {"error": "JSON inválido"}, 400
    text = data.get("text")
    if not text:
        return {"error": "text inválido"}, 400
    if not isinstance(text, str):
        return {"error": "texto vacío"}, 400

    try:
        #text = extract_text_from_file(file)
        X = vectorizer.transform([text])
        prediction = int(svm_model.predict(X)[0])
        prob = svm_model.predict_proba(X)[0][1]

        return jsonify({
            "predicted_label": prediction,
            "prob": prob
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========  CARGAR MODELO roberta =========== #
# Ruta donde Docker guardará el modelo
MODEL_PATH = "/app/model.safetensors"

# ================== CARGAR TOKENIZER ================== #
tokenizer = RobertaTokenizer.from_pretrained("models/roBERTa/tokenizer_tesis")

# ================== CARGAR CONFIG ================== #
config = RobertaConfig.from_pretrained("models/roBERTa/modelo_tesis")

# ================== CARGAR MODELO DESDE SAFETENSORS ================== #
print("Cargando modelo RoBERTa desde model.safetensors...")

state_dict = load_file(MODEL_PATH)

model = RobertaForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=None,
    config=config,
    state_dict=state_dict
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("RoBERTa cargado exitosamente ✔")

#=============       roBERTa       =S============#
@app.route("/predict/roberta", methods=["POST"])
def predictRoBERTa():
    # if "file" not in request.files:
    #     return jsonify({"error": "Debes enviar un archivo"}), 400
    # file = request.files["file"]

    data = request.get_json(silent=True)
    if not data:
        return {"error": "JSON inválido"}, 400
    text = data.get("text")
    if not text:
        return {"error": "text inválido"}, 400
    if not isinstance(text, str):
        return {"error": "texto vacío"}, 400

    try:
        # text = extract_text_from_file(file)
        tokens = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model.to(device)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            logits = model(**tokens).logits
            
        pred = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1)
        prob_pred = probs[0][pred].item()

        return jsonify({"predicted_label": int(pred), "prob": prob_pred})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==== CARGAR MODELO MLP ====
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ==== CARGAR MODELO MLP ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mlp_path = os.path.join(BASE_DIR, "models/mlp/mlp_model.pt")

print("Cargando modelo MLP desde:", mlp_path)
print("Archivos disponibles en BASE_DIR:", os.listdir(BASE_DIR))

mlp_model = SimpleNet(input_dim=384, hidden_dim=256, n_classes=3)

state_dict = torch.load(mlp_path, map_location="cpu")
mlp_model.load_state_dict(state_dict)
mlp_model.eval()

#=============       MLP       =============#
@app.route("/predict/mlp", methods=["POST"])
def predictMLP():
    # if "file" not in request.files:
    #     return jsonify({"error": "Debes enviar un archivo"}), 400

    data = request.get_json(silent=True)
    if not data:
        return {"error": "JSON inválido"}, 400
    text = data.get("text")
    if not text:
        return {"error": "text inválido"}, 400
    if not isinstance(text, str):
        return {"error": "texto vacío"}, 400
    
    try:
        # text = extract_text_from_file(request.files["file"])
        emb = embedder.encode([text], convert_to_numpy=True)

        X = torch.tensor(emb, dtype=torch.float32)

        with torch.no_grad():
            logits = mlp_model(X)
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            prob_pred = float(probs[0][pred].item())

        return jsonify({"predicted_label": int(pred), "prob": prob_pred})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/healthcheck")
def health():
    return {"status": "ok"}

def getData(data):
    data = request.get_json(silent=True)
    if not data:
        return {"error": "JSON inválido"}, 400
    text = data.get("text")
    if not text:
        return {"error": "text inválido"}, 400
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
