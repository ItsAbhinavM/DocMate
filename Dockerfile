FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Set working directory
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
WORKDIR /app/ship_wright
RUN npm install
WORKDIR /app

# Expose port and run backend
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

