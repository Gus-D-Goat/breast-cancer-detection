FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# System libs needed by tensorflow-cpu / opencv-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model.py preprocess.py weights.py README.md ./
COPY Test_images ./Test_images
COPY image ./image

EXPOSE 7860

CMD ["python", "app.py"]
