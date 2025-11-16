from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pdfminer.high_level import extract_text
import docx

app = Flask(__name__)

model = joblib.load("modelo_svm_calibrado.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

def extract_from_pdf(file_path):
    return extract_text(file_path)

def extract_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_file(file):
    filename = file.filename.lower()
    
    tmp_path = "/tmp/" + filename
    file.save(tmp_path)

    if filename.endswith(".pdf"):
        return extract_from_pdf(tmp_path)
    elif filename.endswith(".docx"):
        return extract_from_docx(tmp_path)
    else:  # txt
        return open(tmp_path, "r", encoding="utf-8").read()

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No enviaste archivo"}), 400

    file = request.files["file"]

    try:
        text = extract_text_from_file(file)
        X = vectorizer.transform([text])
        prediction = int(model.predict(X)[0])

        prob = model.predict_proba(X)[0][1]     # probabilidad clase 1 (IA

        return jsonify({
            "predicted_label": prediction,
            "prob": prob
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/healthcheck")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7100)
