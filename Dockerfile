# Reproducible CPU image for the Saint George classifier.
FROM python:3.13-slim

WORKDIR /app

# System deps for Pillow/OpenCV-free image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu

COPY . .

# Default: prepare data + train the baseline, then evaluate.
# Mount the two archives into /data (see README).
CMD ["sh", "-c", "python scripts/prepare_data.py --pos-zip /data/georges.zip --neg-zip /data/non_georges.zip && python scripts/train.py --experiment baseline_resnet18 && python scripts/evaluate.py --experiment baseline_resnet18 --split test"]
