@echo off
REM ========================================
REM Sistema Preditivo de Manutenção v2.0
REM Script de Inicialização
REM ========================================

echo.
echo ========================================
REM  INICIANDO SISTEMA PREDITIVO
REM  DE MANUTENCAO v2.0
echo ========================================
echo.

REM Verificar se ambiente virtual existe
if not exist "venv\Scripts\activate.bat" (
    echo [ERRO] Ambiente virtual nao encontrado!
    echo.
    echo Execute primeiro o arquivo: instalar.bat
    echo.
    pause
    exit /b 1
)

REM Verificar se app.py existe
if not exist "app.py" (
    echo [ERRO] Arquivo app.py nao encontrado!
    echo.
    echo Execute este script na pasta raiz do projeto.
    echo.
    pause
    exit /b 1
)

REM Ativar ambiente virtual
echo [1/2] Ativando ambiente virtual...
call venv\Scripts\activate.bat
echo [OK] Ambiente virtual ativado
echo.

REM Criar diretórios se não existirem
if not exist "outputs\logs" mkdir outputs\logs

REM Verificar se streamlit está instalado
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Streamlit nao encontrado!
    echo.
    echo Execute primeiro: instalar.bat
    echo.
    pause
    exit /b 1
)

echo [OK] Streamlit encontrado
echo.

REM Iniciar aplicação
echo [2/2] Iniciando aplicacao...
echo.
echo ========================================
echo  SISTEMA INICIANDO...
echo ========================================
echo.
echo Aguarde o navegador abrir automaticamente.
echo Se nao abrir, acesse manualmente:
echo.
echo    http://localhost:8501
echo.
echo Para PARAR o sistema:
echo    Pressione Ctrl+C nesta janela
echo.
echo ========================================
echo.

REM Executar Streamlit
streamlit run app.py --server.headless false

REM Se chegou aqui, o Streamlit foi encerrado
echo.
echo Sistema encerrado.
echo.
pause
