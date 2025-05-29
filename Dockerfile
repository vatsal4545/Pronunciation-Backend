FROM python:3.11

# Install system dependencies for Azure and audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libssl-dev \
    libasound2 \
    libasound2-dev \
    build-essential \
    libcurl4 \
    libffi-dev \
    libnss3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["gunicorn", "app:app"] 