# 1. Base image: lightweight Python
FROM python:3.10-slim

# 2. Set work directory inside container
WORKDIR /app

# 3. System deps (for Pillow, torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install Python deps
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the app code
COPY . .

# 6. Expose Streamlit port
EXPOSE 8501

# 7. Streamlit config (to run on 0.0.0.0 inside container)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 8. Run the app
CMD ["streamlit", "run", "app.py"]
