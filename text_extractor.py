import fitz
import uuid
import docx
import os

# Funcion de extraccion de texto
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
