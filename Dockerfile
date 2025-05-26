FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y libsndfile1 libssl-dev libasound2

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["gunicorn", "app:app"] 