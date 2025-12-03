"""
src/data/validators.py
Validação de dados usando schemas Pandera.
"""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from typing import Tuple, Optional, Dict, Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# Schema para dados de falhas
SCHEMA_FALHAS = DataFrameSchema(
    {
        'ativo_unico': Column(str, nullable=False, 
                             description="Identificador único do ativo"),
        'data': Column(pa.DateTime, nullable=False,
                      description="Timestamp da falha"),
        'instalacao': Column(str, nullable=False,
                            description="Localização da instalação"),
        'modulo_envolvido': Column(str, nullable=False,
                                  description="Módulo do sistema"),
        'ativo': Column(str, nullable=False,
                       description="Nome do equipamento"),
        'regional': Column(str, nullable=True,
                         description="Regional operacional"),
        'tipo_manutencao': Column(str, nullable=True,
                                 description="Tipo de manutenção"),
        'descricao': Column(str, nullable=True,
                          description="Descrição do evento"),
        'id_evento': Column(str, nullable=True,
                          description="ID único do evento")
    },
    strict=False,  # Permitir colunas adicionais
    coerce=True    # Converter tipos automaticamente
)


# Schema para features engineered
def create_features_schema(feature_names: list) -> DataFrameSchema:
    """
    Cria schema dinâmico para features engineered.
    
    Args:
        feature_names: Lista de nomes de features
        
    Returns:
        Schema Pandera
    """
    columns = {
        'ativo_unico': Column(str, nullable=False),
        'data': Column(pa.DateTime, nullable=False)
    }
    
    # Adicionar features numéricas
    for feature in feature_names:
        if feature not in ['ativo_unico', 'data']:
            columns[feature] = Column(float, nullable=False,
                                     checks=Check.in_range(-1e10, 1e10))
    
    return DataFrameSchema(columns, strict=False, coerce=True)


def validate_falhas_df(
    df: pd.DataFrame,
    strict: bool = False
) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Valida DataFrame de falhas usando schema Pandera.
    
    Args:
        df: DataFrame a validar
        strict: Se True, levanta exceção em erros; se False, retorna status
        
    Returns:
        Tuple (is_valid, validated_df, errors_dict)
        - is_valid: bool indicando se passou na validação
        - validated_df: DataFrame validado (None se falhou)
        - errors_dict: Dict com erros encontrados
    """
    logger.info("Validando DataFrame de falhas...")
    
    try:
        validated_df = SCHEMA_FALHAS.validate(df, lazy=True)
        logger.info("✓ Validação bem-sucedida")
        return True, validated_df, {}
    
    except pa.errors.SchemaErrors as e:
        logger.warning(f"Erros de validação encontrados: {len(e.failure_cases)} falhas")
        
        errors_dict = {
            'failure_cases': e.failure_cases.to_dict('records') if hasattr(e, 'failure_cases') else [],
            'schema_errors': str(e)
        }
        
        if strict:
            raise
        
        return False, None, errors_dict


def validate_features_df(
    df: pd.DataFrame,
    feature_names: list,
    strict: bool = False
) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Valida DataFrame de features.
    
    Args:
        df: DataFrame a validar
        feature_names: Lista de features esperadas
        strict: Se True, levanta exceção em erros
        
    Returns:
        Tuple (is_valid, validated_df, errors_dict)
    """
    logger.info("Validando DataFrame de features...")
    
    # Verificar se todas as features estão presentes
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        logger.warning(f"Features faltando: {missing_features}")
        errors_dict = {'missing_features': list(missing_features)}
        
        if strict:
            raise ValueError(f"Features faltando: {missing_features}")
        
        return False, None, errors_dict
    
    # Verificar NaNs
    nan_counts = df[feature_names].isnull().sum()
    features_with_nans = nan_counts[nan_counts > 0]
    
    if len(features_with_nans) > 0:
        logger.warning(f"Features com NaN: {features_with_nans.to_dict()}")
        errors_dict = {'features_with_nans': features_with_nans.to_dict()}
        
        if strict:
            raise ValueError(f"Features contêm NaN: {features_with_nans.to_dict()}")
        
        return False, None, errors_dict
    
    # Verificar infinitos
    inf_counts = {}
    for feature in feature_names:
        if df[feature].dtype in [float, int]:
            inf_count = np.isinf(df[feature]).sum()
            if inf_count > 0:
                inf_counts[feature] = inf_count
    
    if inf_counts:
        logger.warning(f"Features com valores infinitos: {inf_counts}")
        errors_dict = {'features_with_inf': inf_counts}
        
        if strict:
            raise ValueError(f"Features contêm infinitos: {inf_counts}")
        
        return False, None, errors_dict
    
    logger.info("✓ Validação de features bem-sucedida")
    return True, df, {}


def validate_model_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = 'classification'
) -> Tuple[bool, str]:
    """
    Valida inputs para treinamento de modelo.
    
    Args:
        X: Features
        y: Target
        task: Tipo de tarefa ('classification' ou 'regression')
        
    Returns:
        Tuple (is_valid, error_message)
    """
    # Tamanhos compatíveis
    if len(X) != len(y):
        return False, f"Tamanhos incompatíveis: X={len(X)}, y={len(y)}"
    
    # Mínimo de amostras
    if len(X) < 10:
        return False, f"Muito poucas amostras: {len(X)} (mínimo 10)"
    
    # Features vazias
    if X.shape[1] == 0:
        return False, "Nenhuma feature fornecida"
    
    # NaNs
    if X.isnull().any().any():
        nan_cols = X.columns[X.isnull().any()].tolist()
        return False, f"Features contêm NaN: {nan_cols}"
    
    if pd.isnull(y).any():
        return False, "Target contém NaN"
    
    # Classificação: mínimo 2 classes
    if task == 'classification':
        n_classes = y.nunique()
        if n_classes < 2:
            return False, f"Target precisa ter pelo menos 2 classes (encontrado: {n_classes})"
    
    return True, ""


import numpy as np  # Importar numpy para uso em validate_features_df
