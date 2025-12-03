#!/bin/bash
# deploy.sh - Script de deploy simplificado

set -e

echo "🚀 Iniciando deploy do Sistema Preditivo de Manutenção..."

# Verificar se Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não encontrado. Instale o Docker primeiro."
    exit 1
fi

# Verificar se Docker Compose está instalado
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose não encontrado. Instale o Docker Compose primeiro."
    exit 1
fi

# Criar .env se não existir
if [ ! -f .env ]; then
    echo "📝 Criando arquivo .env..."
    cp .env.example .env
    echo "⚠️  ATENÇÃO: Configure o arquivo .env com suas API keys antes de prosseguir."
    read -p "Pressione Enter para continuar ou Ctrl+C para cancelar..."
fi

# Criar diretórios necessários
echo "📁 Criando diretórios..."
mkdir -p data/raw data/processed outputs/logs outputs/models outputs/reports

# Build da imagem
echo "🔨 Construindo imagem Docker..."
docker-compose build

# Iniciar serviço
echo "▶️  Iniciando serviço..."
docker-compose up -d

# Aguardar health check
echo "⏳ Aguardando aplicação inicializar..."
sleep 10

# Verificar status
if docker-compose ps | grep -q "Up"; then
    echo "✅ Deploy concluído com sucesso!"
    echo ""
    echo "🌐 Acesse a aplicação em: http://localhost:8501"
    echo ""
    echo "📊 Comandos úteis:"
    echo "  - Ver logs:        docker-compose logs -f"
    echo "  - Parar serviço:   docker-compose down"
    echo "  - Reiniciar:       docker-compose restart"
    echo "  - Ver status:      docker-compose ps"
else
    echo "❌ Falha no deploy. Verifique os logs:"
    docker-compose logs
    exit 1
fi
