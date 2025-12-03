"""
src/data/preprocessors.py
Funções de pré-processamento e limpeza de dados.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'zero',
    numeric_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Trata valores faltantes em colunas numéricas.
    
    Args:
        df: DataFrame com valores faltantes
        strategy: Estratégia ('zero', 'mean', 'median', 'forward_fill')
        numeric_cols: Colunas a processar (None = todas numéricas)
        
    Returns:
        DataFrame com valores faltantes tratados
    """
    df = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Tratando valores faltantes (estratégia: {strategy})")
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue
        
        if strategy == 'zero':
            df[col] = df[col].fillna(0)
        elif strategy == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif strategy == 'forward_fill':
            df[col] = df[col].fillna(method='ffill')
        else:
            logger.warning(f"Estratégia desconhecida: {strategy}, usando zero")
            df[col] = df[col].fillna(0)
        
        logger.info(f"  {col}: {missing_count} valores preenchidos")
    
    return df


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Remove outliers de colunas numéricas.
    
    Args:
        df: DataFrame original
        columns: Colunas a processar (None = todas numéricas)
        method: Método de detecção ('zscore', 'iqr')
        threshold: Threshold para outliers (zscore: ±3, iqr: 1.5)
        
    Returns:
        DataFrame sem outliers
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    initial_len = len(df)
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'zscore':
            # Z-score method
            mean = df[col].mean()
            std = df[col].std()
            z_scores = np.abs((df[col] - mean) / (std + 1e-10))
            mask &= (z_scores <= threshold)
        
        elif method == 'iqr':
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
    
    df_filtered = df[mask].copy()
    removed_count = initial_len - len(df_filtered)
    
    if removed_count > 0:
        logger.info(f"Outliers removidos: {removed_count} linhas ({removed_count/initial_len*100:.1f}%)")
    
    return df_filtered


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'onehot'
) -> pd.DataFrame:
    """
    Codifica variáveis categóricas.
    
    Args:
        df: DataFrame original
        columns: Colunas categóricas a codificar
        method: Método ('onehot', 'label')
        
    Returns:
        DataFrame com variáveis codificadas
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
            logger.info(f"One-hot encoding: {col} -> {len(dummies.columns)} colunas")
        
        elif method == 'label':
            # Label encoding
            df[col] = pd.factorize(df[col])[0]
            logger.info(f"Label encoding: {col}")
    
    return df


def scale_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'standard'
) -> tuple[pd.DataFrame, Any]:
    """
    Escala features numéricas.
    
    Args:
        df: DataFrame original
        columns: Colunas a escalar (None = todas numéricas)
        method: Método ('standard', 'robust', 'minmax')
        
    Returns:
        Tuple (df_scaled, scaler) - DataFrame escalado e scaler fitted
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    df[columns] = scaler.fit_transform(df[columns])
    
    logger.info(f"Features escaladas ({method}): {len(columns)} colunas")
    
    return df, scaler


def remove_duplicate_rows(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Remove linhas duplicadas.
    
    Args:
        df: DataFrame original
        subset: Colunas para verificar duplicatas (None = todas)
        keep: Qual duplicata manter ('first', 'last', False)
        
    Returns:
        DataFrame sem duplicatas
    """
    initial_len = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep).copy()
    removed_count = initial_len - len(df)
    
    if removed_count > 0:
        logger.info(f"Duplicatas removidas: {removed_count} linhas")
    
    return df


def filter_by_date_range(
    df: pd.DataFrame,
    date_column: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Filtra DataFrame por intervalo de datas.
    
    Args:
        df: DataFrame original
        date_column: Nome da coluna de data
        start_date: Data inicial (None = sem limite inferior)
        end_date: Data final (None = sem limite superior)
        
    Returns:
        DataFrame filtrado
    """
    df = df.copy()
    
    if start_date:
        df = df[df[date_column] >= start_date]
    
    if end_date:
        df = df[df[date_column] <= end_date]
    
    logger.info(f"Filtrado por data: {len(df)} linhas restantes")
    
    return df


def aggregate_by_ativo(
    df: pd.DataFrame,
    agg_dict: Dict[str, Any]
) -> pd.DataFrame:
    """
    Agrega dados por ativo_unico.
    
    Args:
        df: DataFrame original
        agg_dict: Dicionário de agregações {coluna: função}
        
    Returns:
        DataFrame agregado
        
    Example:
        >>> df_agg = aggregate_by_ativo(df, {
        ...     'tbf': ['mean', 'std', 'min', 'max'],
        ...     'falhas_acumuladas': 'max'
        ... })
    """
    df_agg = df.groupby('ativo_unico').agg(agg_dict).reset_index()
    
    # Flatten multi-level columns se necessário
    if isinstance(df_agg.columns, pd.MultiIndex):
        df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]
    
    logger.info(f"Agregado por ativo: {len(df_agg)} ativos únicos")
    
    return df_agg
