# Usar una imagen base oficial de Python
FROM python:3.9-slim
# Establecer el directorio de trabajo
WORKDIR /app
# Copiar el archivo de requerimientos
COPY requirements.txt .
# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt
# Copiar todo el código de la aplicación (app.py)
COPY . .
# Exponer el puerto en el que corre Flask
EXPOSE 1100
# Comando para ejecutar la aplicación
CMD ["python", "app.py"]