@echo off
REM ========================================
REM Sistema Preditivo de Manutenção v2.0
REM Script de Instalação Automática
REM ========================================

echo.
echo ========================================
echo  INSTALACAO DO SISTEMA PREDITIVO
echo  DE MANUTENCAO v2.0
echo ========================================
echo.

REM Verificar se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado!
    echo.
    echo Por favor, instale Python 3.9+ antes de continuar:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Python encontrado:
python --version
echo.

REM Verificar se está no diretório correto
if not exist "requirements.txt" (
    echo [ERRO] Arquivo requirements.txt nao encontrado!
    echo.
    echo Execute este script na pasta raiz do projeto.
    echo.
    pause
    exit /b 1
)

echo [OK] Arquivo requirements.txt encontrado
echo.

REM Criar ambiente virtual se não existir
if not exist "venv" (
    echo [1/4] Criando ambiente virtual...
    python -m venv venv
    if errorlevel 1 (
        echo [ERRO] Falha ao criar ambiente virtual
        pause
        exit /b 1
    )
    echo [OK] Ambiente virtual criado
) else (
    echo [OK] Ambiente virtual ja existe
)
echo.

REM Ativar ambiente virtual
echo [2/4] Ativando ambiente virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERRO] Falha ao ativar ambiente virtual
    pause
    exit /b 1
)
echo [OK] Ambiente virtual ativado
echo.

REM Atualizar pip
echo [3/4] Atualizando pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [AVISO] Falha ao atualizar pip, continuando...
)
echo [OK] pip atualizado
echo.

REM Instalar dependências
echo [4/4] Instalando dependencias (isso pode demorar alguns minutos)...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERRO] Falha ao instalar algumas dependencias
    echo.
    echo Tente executar manualmente:
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  INSTALACAO CONCLUIDA COM SUCESSO!
echo ========================================
echo.
echo Proximos passos:
echo.
echo 1. Configure suas API keys (opcional):
echo    - Copie .env.example para .env
echo    - Adicione sua GEMINI_API_KEY
echo.
echo 2. Execute o sistema:
echo    - Clique duas vezes em iniciar.bat
echo    - Ou execute: streamlit run app.py
echo.
echo 3. Acesse no navegador:
echo    - http://localhost:8501
echo.
echo ========================================
echo.

REM Criar diretórios necessários
echo Criando diretorios de dados...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "outputs\logs" mkdir outputs\logs
if not exist "outputs\models" mkdir outputs\models
if not exist "outputs\reports" mkdir outputs\reports
echo [OK] Diretorios criados
echo.

echo Instalacao finalizada!
echo.
pause
