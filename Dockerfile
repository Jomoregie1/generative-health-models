# ------------ Base image ------------
FROM python:3.11-slim

# Fast, quiet pip; expose Streamlit port via env
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501

# ------------ OS packages ------------
# curl/unzip for downloading the bundle; libgl for matplotlib; certs for HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      unzip \
      git \
      libglib2.0-0 \
      libgl1 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ------------ Workdir ------------
WORKDIR /app

# ------------ Python deps ------------
# If torch is already in requirements.txt, remove the three lines that install torch here.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 && \
    pip install --no-cache-dir -r requirements.txt

# ------------ App code ------------
COPY src/ ./src/

# ------------ Bootstrap (auto-download & launch) ------------
# Make sure you created docker/bootstrap.sh in your repo
COPY docker/bootstrap.sh /app/docker/bootstrap.sh

# Optional: normalize CRLF->LF if you're editing on Windows and hit shell errors
# RUN apt-get update && apt-get install -y --no-install-recommends dos2unix && \
#     dos2unix /app/docker/bootstrap.sh && \
#     rm -rf /var/lib/apt/lists/*

RUN chmod +x /app/docker/bootstrap.sh

# ------------ Expose & entrypoint ------------
EXPOSE 8501
ENTRYPOINT ["/app/docker/bootstrap.sh"]