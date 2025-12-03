"""
cli_train.py
CLI para treinamento de modelos preditivos.

Uso:
    python cli_train.py --data data/raw/falhas.xlsx --version v20251203
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logging_config import setup_logging, load_config
from src.data.loaders import load_falhas_excel
from src.features.engineering import FeatureEngineeringPipeline
from src.features.target_builder import create_multi_horizon_targets, create_temporal_splits
from src.models.trainer import ModelTrainer
from src.utils.io import save_model_artifacts

logger = setup_logging(Path("outputs/logs/training.log"))


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Treinamento de modelos preditivos")
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Caminho do arquivo Excel de falhas'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Nome da versão (padrão: timestamp automático)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Caminho do arquivo de configuração'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Nível de logging'
    )
    
    return parser.parse_args()


def main():
    """Função principal de treinamento."""
    args = parse_args()
    
    logger.info("="*70)
    logger.info("INICIANDO RETREINAMENTO DE MODELOS PREDITIVOS")
    logger.info("="*70)
    
    # Carregar configuração
    logger.info(f"Carregando configuração: {args.config}")
    config = load_config(Path(args.config))
    logger.info(f"Projeto: {config['project']['name']} v{config['project']['version']}")
    
    # 1. Carregar dados
    logger.info(f"\n{'='*70}")
    logger.info("ETAPA 1: CARREGAMENTO DE DADOS")
    logger.info(f"{'='*70}")
    
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Arquivo não encontrado: {data_path}")
        sys.exit(1)
    
    df_raw = load_falhas_excel(data_path, config)
    logger.info(f"✓ Dados carregados: {len(df_raw)} registros, {df_raw['ativo_unico'].nunique()} ativos")
    
    # 2. Feature Engineering
    logger.info(f"\n{'='*70}")
    logger.info("ETAPA 2: FEATURE ENGINEERING")
    logger.info(f"{'='*70}")
    
    pipeline = FeatureEngineeringPipeline(
        lags=config['features']['lags'],
        rolling_windows=config['features']['rolling_windows'],
        include_sazonalidade=config['features']['include_sazonalidade'],
        include_interacoes=config['features']['include_interacoes']
    )
    
    df_features = pipeline.fit_transform(df_raw)
    feature_names = pipeline.get_feature_names()
    logger.info(f"✓ Features criadas: {len(feature_names)}")
    
    # 3. Criar targets
    logger.info(f"\n{'='*70}")
    logger.info("ETAPA 3: CRIAÇÃO DE TARGETS")
    logger.info(f"{'='*70}")
    
    horizontes = config['models']['horizontes']
    df_features = create_multi_horizon_targets(df_features, horizontes)
    
    # 4. Splits temporais
    logger.info(f"\n{'='*70}")
    logger.info("ETAPA 4: SPLITS TEMPORAIS")
    logger.info(f"{'='*70}")
    
    mask_train, mask_val, mask_test = create_temporal_splits(
        df_features,
        frac_train=config['models']['split']['train'],
        frac_val=config['models']['split']['val']
    )
    
    # 5. Treinamento
    logger.info(f"\n{'='*70}")
    logger.info("ETAPA 5: TREINAMENTO DE MODELOS")
    logger.info(f"{'='*70}")
    
    trainer = ModelTrainer(config)
    results = trainer.train_all_horizons(
        df_features,
        feature_names,
        horizontes,
        mask_train,
        mask_val,
        mask_test
    )
    
    # 6. Salvar artefatos
    logger.info(f"\n{'='*70}")
    logger.info("ETAPA 6: SALVAMENTO DE ARTEFATOS")
    logger.info(f"{'='*70}")
    
    version_path = save_model_artifacts(
        results,
        feature_names,
        config,
        version=args.version
    )
    
    # Sumário final
    logger.info(f"\n{'='*70}")
    logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO! ✓")
    logger.info(f"{'='*70}")
    logger.info(f"Versão salva: {version_path}")
    logger.info(f"Horizontes treinados: {list(results.keys())}")
    
    logger.info("\nMétricas dos Campeões (Validação):")
    for h, result in results.items():
        logger.info(
            f"  {h}d: {result['model_name']} - "
            f"F1={result['metrics_val']['F1-Score']:.3f}, "
            f"AUC={result['metrics_val'].get('AUC-ROC', 0):.3f}"
        )
    
    logger.info("\nPara usar os modelos treinados:")
    logger.info(f"  streamlit run app.py")
    logger.info(f"  ou python cli_inference.py --models {version_path}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Erro durante treinamento: {e}", exc_info=True)
        sys.exit(1)
