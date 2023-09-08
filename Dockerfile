
# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Sets environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container to /app
ENV APP_HOME /app
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Run app.py when the container launches
CMD exec uvicorn --host 0.0.0.0 --port 8080  main:app

