# deploy.ps1 - Script de deploy para Windows

Write-Host "🚀 Iniciando deploy do Sistema Preditivo de Manutenção..." -ForegroundColor Green

# Verificar se Docker está instalado
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker não encontrado. Instale o Docker Desktop primeiro." -ForegroundColor Red
    exit 1
}

# Verificar se Docker Compose está disponível
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker Compose não encontrado. Instale o Docker Compose primeiro." -ForegroundColor Red
    exit 1
}

# Criar .env se não existir
if (-not (Test-Path .env)) {
    Write-Host "📝 Criando arquivo .env..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "⚠️  ATENÇÃO: Configure o arquivo .env com suas API keys antes de prosseguir." -ForegroundColor Yellow
    Read-Host "Pressione Enter para continuar ou Ctrl+C para cancelar"
}

# Criar diretórios necessários
Write-Host "📁 Criando diretórios..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path data/raw, data/processed, outputs/logs, outputs/models, outputs/reports | Out-Null

# Build da imagem
Write-Host "🔨 Construindo imagem Docker..." -ForegroundColor Cyan
docker-compose build

# Iniciar serviço
Write-Host "▶️  Iniciando serviço..." -ForegroundColor Cyan
docker-compose up -d

# Aguardar health check
Write-Host "⏳ Aguardando aplicação inicializar..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Verificar status
$status = docker-compose ps --format json | ConvertFrom-Json
if ($status.State -eq "running") {
    Write-Host "✅ Deploy concluído com sucesso!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Acesse a aplicação em: http://localhost:8501" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📊 Comandos úteis:" -ForegroundColor Yellow
    Write-Host "  - Ver logs:        docker-compose logs -f"
    Write-Host "  - Parar serviço:   docker-compose down"
    Write-Host "  - Reiniciar:       docker-compose restart"
    Write-Host "  - Ver status:      docker-compose ps"
} else {
    Write-Host "❌ Falha no deploy. Verifique os logs:" -ForegroundColor Red
    docker-compose logs
    exit 1
}
