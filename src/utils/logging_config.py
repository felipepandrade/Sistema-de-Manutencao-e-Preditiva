"""
src/utils/logging_config.py
Configuração centralizada de logging para o sistema.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import yaml


def setup_logging(
    log_file: Optional[Path] = None,
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configura sistema de logging com rotação de arquivos.
    
    Args:
        log_file: Caminho do arquivo de log (None = apenas console)
        log_level: Nível de logging (DEBUG, INFO, WARNING, ERROR)
        log_format: Formato customizado de log
        max_bytes: Tamanho máximo do arquivo antes de rotacionar
        backup_count: Quantidade de arquivos de backup a manter
        
    Returns:
        Logger configurado
        
    Example:
        >>> logger = setup_logging(Path("outputs/logs/app.log"))
        >>> logger.info("Sistema iniciado")
    """
    # Formato padrão
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configurar logger raiz
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remover handlers existentes
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler com rotação (se especificado)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: Path = None) -> dict:
    """
    Carrega configuração do arquivo YAML.
    
    Args:
        config_path: Caminho do arquivo config.yaml (None = usar padrão)
        
    Returns:
        Dicionário com configurações
    """
    if config_path is None:
        # Buscar config.yaml na raiz do projeto
        current_dir = Path(__file__).parent
        config_path = current_dir.parent.parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_logger(name: str) -> logging.Logger:
    """
    Obtém logger para módulo específico.
    
    Args:
        name: Nome do módulo (use __name__)
        
    Returns:
        Logger configurado
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processando dados...")
    """
    return logging.getLogger(name)
