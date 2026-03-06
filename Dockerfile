FROM python:3.9-slim

WORKDIR /app

# Install dependencies sistem
RUN apt-get update && apt-get install -y build-essential

# Copy dan Install Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh project
COPY . .

# Install folder src sebagai package (Ini langkah krusial!)
RUN pip install -e .

# Command default
CMD ["python", "entrypoint/run_train.py"]