FROM python:3.11-slim

WORKDIR /app

# Install system dependencies with better compatibility
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-hin \
    pkg-config \
    libhdf5-dev \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Install packages in stages to avoid conflicts
COPY requirements.txt .
RUN pip install --no-cache-dir --no-deps torch==2.1.0
RUN pip install --no-cache-dir --no-deps numpy==1.24.4
RUN pip install --no-cache-dir --no-deps librosa==0.11.0
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Create data directory
RUN mkdir -p data

# Expose port for Koyeb
EXPOSE 8080

# Run your bot
CMD ["python", "app.py"]
