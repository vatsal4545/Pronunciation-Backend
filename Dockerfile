FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libssl-dev \
    libasound2 \
    libasound2-dev \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["gunicorn", "app:app"] 