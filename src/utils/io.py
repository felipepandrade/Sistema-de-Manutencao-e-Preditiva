"""
src/utils/io.py
Funções de I/O para serialização e carregamento de modelos.
"""

import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import shutil

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def save_model_artifacts(
    models_by_horizon: Dict[int, Dict[str, Any]],
    feature_names: List[str],
    config: Dict[str, Any],
    version: Optional[str] = None,
    models_dir: Path = None
) -> Path:
    """
    Salva artefatos de modelos com versionamento.
    
    Args:
        models_by_horizon: Dicionário {horizonte: {model, calibrador, métricas, ...}}
        feature_names: Lista de nomes de features usadas
        config: Configuração do sistema
        version: Nome da versão (None = timestamp automático)
        models_dir: Diretório base para modelos
        
    Returns:
        Path para diretório da versão salva
        
    Example:
        >>> path = save_model_artifacts(results, features, config)
        >>> print(f"Modelos salvos em: {path}")
    """
    if models_dir is None:
        models_dir = Path(config['paths']['models'])
    
    # Criar nome de versão
    if version is None:
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    version_path = models_dir / version
    version_path.mkdir(parents=True, exist_ok=True)
    
    # Salvar modelos e calibradores por horizonte
    for h, result in models_by_horizon.items():
        # Modelo
        model_file = version_path / f"modelo_{h}d.pkl"
        joblib.dump(result['model'], model_file)
        logger.info(f"Modelo {h}d salvo: {model_file}")
        
        # Calibrador (se existir)
        if result.get('calibrador'):
            calibrador_file = version_path / f"calibrador_{h}d.pkl"
            joblib.dump(result['calibrador'], calibrador_file)
            logger.info(f"Calibrador {h}d salvo: {calibrador_file}")
    
    # Salvar metadados
    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'features': feature_names,
        'n_features': len(feature_names),
        'models_by_horizon': {}
    }
    
    for h, result in models_by_horizon.items():
        metadata['models_by_horizon'][str(h)] = {
            'model_name': result.get('model_name', 'Unknown'),
            'best_threshold': float(result.get('best_threshold', 0.5)),
            'metrics_val': {k: float(v) for k, v in result.get('metrics_val', {}).items()},
            'metrics_test': {k: float(v) for k, v in result.get('metrics_test', {}).items()}
        }
    
    metadata_file = version_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Metadados salvos: {metadata_file}")
    
    # Criar/atualizar symlink 'latest'
    # Criar/atualizar symlink 'latest'
    latest_path = models_dir / "latest"
    
    # Função auxiliar para remover diretório com retry
    def remove_readonly(func, path, excinfo):
        import os
        import stat
        os.chmod(path, stat.S_IWRITE)
        func(path)
        
    if latest_path.exists():
        try:
            if latest_path.is_symlink():
                latest_path.unlink()
            else:
                shutil.rmtree(latest_path, onerror=remove_readonly)
        except Exception as e:
            logger.warning(f"Não foi possível remover 'latest' antigo: {e}")
            # Tentar renomear se não der para remover (workaround Windows)
            try:
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                latest_path.rename(models_dir / f"latest_old_{timestamp}")
            except Exception as e2:
                logger.error(f"Falha crítica ao limpar 'latest': {e2}")

    try:
        # Windows requer privilégios para symlinks, usar cópia como fallback
        # Tentar criar symlink apenas se tiver permissão
        try:
            latest_path.symlink_to(version, target_is_directory=True)
            logger.info(f"Symlink 'latest' criado -> {version}")
        except (OSError, AttributeError):
            # Fallback: copiar diretório
            shutil.copytree(version_path, latest_path)
            logger.info(f"Symlink não suportado/permitido, diretório copiado para 'latest'")
    except Exception as e:
        logger.error(f"Erro ao criar 'latest': {e}")
    
    logger.info(f"✓ Artefatos salvos com sucesso: {version_path}")
    
    return version_path


def load_model_artifacts(
    version_path: Path,
    horizontes: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Carrega artefatos de modelos de uma versão.
    
    Args:
        version_path: Caminho para diretório da versão
        horizontes: Lista de horizontes a carregar (None = todos)
        
    Returns:
        Dict com:
        - models: Dict[int, Dict] com modelo e calibrador por horizonte
        - metadata: Dict com metadados
        - feature_names: List[str] com nomes de features
        
    Raises:
        FileNotFoundError: Se versão não existir
    """
    if not version_path.exists():
        raise FileNotFoundError(f"Versão não encontrada: {version_path}")
    
    # Carregar metadados
    metadata_file = version_path / "metadata.json"
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    feature_names = metadata['features']
    
    # Determinar horizontes a carregar
    if horizontes is None:
        horizontes = [int(h) for h in metadata['models_by_horizon'].keys()]
    
    # Carregar modelos
    models = {}
    for h in horizontes:
        model_file = version_path / f"modelo_{h}d.pkl"
        calibrador_file = version_path / f"calibrador_{h}d.pkl"
        
        if not model_file.exists():
            logger.warning(f"Modelo {h}d não encontrado, pulando...")
            continue
        
        models[h] = {
            'model': joblib.load(model_file),
            'calibrador': joblib.load(calibrador_file) if calibrador_file.exists() else None,
            'threshold': metadata['models_by_horizon'][str(h)]['best_threshold']
        }
        
        logger.info(f"Modelo {h}d carregado: {model_file}")
    
    logger.info(f"✓ Artefatos carregados: {len(models)} horizontes")
    
    return {
        'models': models,
        'metadata': metadata,
        'feature_names': feature_names
    }


def get_available_memory_gb() -> float:
    """
    Retorna memória RAM disponível em GB.
    
    Returns:
        Memória disponível em GB
    """
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        logger.warning("psutil não disponível, assumindo 4GB")
        return 4.0
    except Exception as e:
        logger.warning(f"Erro ao obter memória: {e}, assumindo 4GB")
        return 4.0
