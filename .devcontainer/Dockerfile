# Use the official Python image from the Docker Hub
FROM python:3.8

# Install essential libraries
RUN apt-get update && apt-get install -y \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Install Python dependencies
COPY server/requirements.txt client/requirements.txt ./
RUN pip install -r requirements.txt

# Add a verification command
RUN echo "Dockerfile is being executed" > /workspace/dockerfile_check.txt