FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PORT=8501
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl wget unzip git libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # install CPU torch if it's NOT in requirements.txt; otherwise remove next 3 lines
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio && \
    pip install --no-cache-dir -r requirements.txt

# code
COPY src/ ./src/

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]