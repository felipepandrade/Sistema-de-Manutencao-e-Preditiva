"""
cli_inference.py
CLI para inferência em lote usando modelos treinados.

Uso:
    python cli_inference.py --input data/raw/novos_dados.xlsx --models models/latest --output outputs/predictions/resultado.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logging_config import setup_logging, load_config
from src.data.loaders import load_falhas_excel
from src.features.engineering import FeatureEngineeringPipeline
from src.inference.predictor import PredictorPipeline

logger = setup_logging(Path("outputs/logs/inference.log"))


def parse_args():
    parser = argparse.ArgumentParser(description="Inferência em lote")
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Arquivo Excel com novos dados'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='models/latest',
        help='Diretório da versão do modelo (ex: models/latest)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Arquivo de saída CSV (opcional)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Arquivo de configuração'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("="*60)
    logger.info("INICIANDO INFERÊNCIA EM LOTE")
    logger.info("="*60)
    
    # Carregar config
    config = load_config(Path(args.config))
    
    # 1. Carregar dados
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Arquivo de entrada não encontrado: {input_path}")
        sys.exit(1)
        
    logger.info(f"Carregando dados: {input_path}")
    df_raw = load_falhas_excel(input_path, config)
    logger.info(f"✓ Dados carregados: {len(df_raw)} registros")
    
    # 2. Feature Engineering
    logger.info("Gerando features...")
    pipeline = FeatureEngineeringPipeline(
        lags=config['features']['lags'],
        rolling_windows=config['features']['rolling_windows'],
        include_sazonalidade=config['features']['include_sazonalidade'],
        include_interacoes=config['features']['include_interacoes']
    )
    
    df_features = pipeline.fit_transform(df_raw)
    logger.info(f"✓ Features geradas: {df_features.shape[1]} colunas")
    
    # 3. Predição
    models_path = Path(args.models)
    version_name = models_path.name
    
    logger.info(f"Carregando modelos de: {models_path}")
    
    try:
        predictor = PredictorPipeline(model_version=version_name, config=config)
        df_predictions = predictor.predict(df_features)
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        sys.exit(1)
        
    # 4. Salvar resultados
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(f"outputs/predictions/predicoes_{timestamp}.csv")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_predictions.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"✓ Resultados salvos em: {output_path}")
    
    # Exibir resumo
    logger.info("\nResumo de Alto Risco (30 dias):")
    high_risk = df_predictions[df_predictions['Classe_30d'] == 'Alto Risco']
    logger.info(f"  Total: {len(high_risk)} ativos")
    if not high_risk.empty:
        logger.info("  Top 5:")
        for _, row in high_risk.head(5).iterrows():
            logger.info(f"    - {row['ativo_unico']}: {row['ProbML_Media_30d']:.1%}")


if __name__ == '__main__':
    main()
