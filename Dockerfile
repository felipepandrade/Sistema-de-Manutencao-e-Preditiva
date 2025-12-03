# Dockerfile
# Sistema Preditivo de Manutenção - Optimized Multi-stage Build

# Stage 1: Base com dependências do sistema
FROM python:3.10-slim as base

# Metadata
LABEL maintainer="Sistema Preditivo de Manutenção"
LABEL version="2.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependências Python
FROM base as dependencies

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: Aplicação
FROM dependencies as application

WORKDIR /app

# Copiar código fonte
COPY . .

# Criar diretórios necessários
RUN mkdir -p outputs/logs outputs/models outputs/reports data/raw data/processed

# Expor porta do Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando padrão
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
