# Imagen base
FROM python:3.9-slim

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    wget \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelo desde Github Releases
RUN wget -O /app/model.safetensors \
    https://github.com/LogiqueNoire/detectorIABack/releases/download/v1.0/model.safetensors

# Descargar modelo de embeddings dentro del contenedor
RUN python3 - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
EOF

# Copiar el proyecto
COPY . .

# Exponer puerto
EXPOSE 5000

# Ejecutar Flask
CMD ["python3", "app.py"]