# Docker OLLAMA Setup Guide

## Option 1: Run Only OLLAMA in Docker (Recommended for Development)

### 1. Start OLLAMA Container
```bash
# Basic setup
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  -e OLLAMA_ORIGINS=* \
  ollama/ollama

# With GPU support (if you have NVIDIA GPU)
docker run -d \
  --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  -e OLLAMA_ORIGINS=* \
  ollama/ollama
```

### 2. Download Models
```bash
# Download models inside the container
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama pull mistral:7b
docker exec -it ollama ollama pull qwen2.5:7b

# List installed models
docker exec -it ollama ollama list
```

### 3. Run Your App Locally
```bash
# Your app runs locally, connects to Docker OLLAMA
export OLLAMA_BASE_URL=http://localhost:11434
streamlit run app.py
```

---

## Option 2: Run Both OLLAMA and Your App in Docker

### 1. Create Environment File (.env)
```bash
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
OLLAMA_BASE_URL=http://ollama:11434
```

### 2. Start Services with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Download Models
```bash
# Download models after OLLAMA is running
docker-compose exec ollama ollama pull llama3.1:8b
docker-compose exec ollama ollama pull mistral:7b
docker-compose exec ollama ollama pull qwen2.5:7b
```

### 4. Access Your App
- App: http://localhost:8501
- OLLAMA API: http://localhost:11434

---

## Option 3: Standalone OLLAMA with Docker Compose

### Create simple docker-compose.yml:
```yaml
version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_ORIGINS=*
    restart: unless-stopped

volumes:
  ollama_data:
```

```bash
# Start OLLAMA
docker-compose up -d ollama

# Download models
docker-compose exec ollama ollama pull llama3.1:8b

# Run your app locally
streamlit run app.py
```

---

## Model Management Commands

### Download Popular Models:
```bash
# Small models (4-8GB RAM)
docker exec -it ollama ollama pull llama3.2:3b
docker exec -it ollama ollama pull mistral:7b

# Medium models (8-16GB RAM)  
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama pull qwen2.5:7b

# Large models (16GB+ RAM)
docker exec -it ollama ollama pull llama3.1:70b
docker exec -it ollama ollama pull qwen2.5:14b
```

### Model Management:
```bash
# List models
docker exec -it ollama ollama list

# Remove model
docker exec -it ollama ollama rm model_name

# Check model info
docker exec -it ollama ollama show llama3.1:8b
```

---

## Troubleshooting

### OLLAMA Container Issues:
```bash
# Check container status
docker ps

# View OLLAMA logs
docker logs ollama

# Restart OLLAMA
docker restart ollama

# Check OLLAMA health
curl http://localhost:11434/api/version
```

### Memory Issues:
```bash
# Monitor container resources
docker stats ollama

# Limit container memory (if needed)
docker run -d \
  --name ollama \
  --memory=8g \
  --memory-swap=12g \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama
```

### GPU Issues:
```bash
# Check GPU availability
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# OLLAMA with specific GPU
docker run -d \
  --name ollama \
  --gpus '"device=0"' \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama
```

---

## Performance Tips

1. **Use GPU acceleration** if available for faster inference
2. **Choose appropriate model sizes** based on your hardware
3. **Use Docker volumes** to persist models between container restarts
4. **Monitor resource usage** with `docker stats`
5. **Set memory limits** to prevent system overload

---

## Environment Variables

Set these in your shell or `.env` file:

```bash
# For Docker OLLAMA
export OLLAMA_BASE_URL=http://localhost:11434

# For Docker Compose setup
export OLLAMA_BASE_URL=http://ollama:11434

# Enable OLLAMA in your app
export USE_OLLAMA=true
export OLLAMA_MODEL=llama3.1:8b
```