
FROM python:3.8-slim
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY app /app
CMD ["flask", "run", "--host=0.0.0.0"]
