# Base image with Python 3.9
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    langchain_community \
    pypdf \
    llama-cpp-python \
    transformers \
    accelerate \
    langchain \
    faiss-cpu \
    sentence-transformers \
    bitsandbytes

# Copy your application code
COPY main.py .
COPY input/ ./input/

# Command to run the application
CMD ["python", "main.py"]