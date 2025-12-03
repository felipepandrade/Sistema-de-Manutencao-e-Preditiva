"""
scheduler.py
Script de agendamento para automação de retreinamento e relatórios.
"""

import schedule
import time
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from src.utils.logging_config import setup_logging

# Configurar logs específicos para o scheduler
logger = setup_logging(Path("outputs/logs/scheduler.log"))


def run_training_job():
    """Executa o job de treinamento."""
    logger.info("Iniciando job de retreinamento agendado...")
    
    try:
        # Comando para executar cli_train.py
        # Assumindo que o arquivo de dados é sempre o mesmo ou atualizado
        data_path = "data/raw/falhas.xlsx"
        
        if not Path(data_path).exists():
            logger.error(f"Arquivo de dados não encontrado: {data_path}")
            return
            
        cmd = [sys.executable, "cli_train.py", "--data", data_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✓ Retreinamento concluído com sucesso.")
            logger.info(result.stdout)
        else:
            logger.error("❌ Falha no retreinamento.")
            logger.error(result.stderr)
            
    except Exception as e:
        logger.error(f"Erro ao executar job de treinamento: {e}")


def run_report_job():
    """Gera relatório periódico."""
    logger.info("Iniciando geração de relatório agendado...")
    # Implementar lógica para gerar relatório usando ReportGenerator
    # Isso exigiria carregar dados e predições primeiro
    pass


def main():
    logger.info("="*60)
    logger.info("SCHEDULER INICIADO")
    logger.info("="*60)
    
    # Configurar agendamentos
    # Exemplo: Retreinar todo domingo às 02:00
    schedule.every().sunday.at("02:00").do(run_training_job)
    
    # Exemplo: Retreinar a cada 3 dias (para teste)
    # schedule.every(3).days.do(run_training_job)
    
    logger.info("Agendamentos configurados:")
    logger.info("- Retreinamento: Todo Domingo às 02:00")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Verificar a cada minuto
    except KeyboardInterrupt:
        logger.info("Scheduler interrompido pelo usuário.")
    except Exception as e:
        logger.error(f"Erro fatal no scheduler: {e}")


if __name__ == "__main__":
    main()
