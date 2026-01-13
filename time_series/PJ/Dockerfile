FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Minimal deps for matplotlib + tensorflow runtime on slim
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     libstdc++6     libgomp1     libgl1     libglib2.0-0     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

CMD ["python", "-m", "app.bot"]
