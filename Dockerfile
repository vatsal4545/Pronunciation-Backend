FROM python:3.11

# Install system dependencies (removed Azure-specific libs)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["gunicorn", "app:app"] 