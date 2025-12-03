"""
src/features/target_builder.py
Criação de alvos multi-horizonte e splits temporais.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_multi_horizon_targets(
    df: pd.DataFrame,
    horizontes: List[int] = [3, 7, 15, 30]
) -> pd.DataFrame:
    """
    Cria alvos binários multi-horizonte: falha ocorrerá em N dias?
    
    Estratégia:
    - Para cada evento de falha, calcular dias até a próxima falha
    - Se próxima falha <= horizonte dias → target=1, senão target=0
    
    Args:
        df: DataFrame com colunas 'ativo_unico' e 'data'
        horizontes: Lista de horizontes em dias (ex: [3, 7, 15, 30])
        
    Returns:
        DataFrame com colunas target adicionadas: falha_em_3d, falha_em_7d, etc.
        
    Example:
        >>> df_with_targets = create_multi_horizon_targets(df, [7, 30])
        >>> print(df_with_targets[['ativo_unico', 'data', 'falha_em_7d', 'falha_em_30d']])
    """
    logger.info(f"Criando targets para horizontes: {horizontes}")
    
    df = df.sort_values(by=['ativo_unico', 'data']).copy()
    
    # Próxima data de falha (shift -1 dentro de cada grupo)
    df['proxima_data'] = df.groupby('ativo_unico')['data'].shift(-1)
    df['dias_ate_proxima'] = (df['proxima_data'] - df['data']).dt.days
    
    # Criar alvos binários para cada horizonte
    for h in horizontes:
        target_col = f'falha_em_{h}d'
        df[target_col] = np.where(
            (df['dias_ate_proxima'].notna()) & (df['dias_ate_proxima'] <= h),
            1, 0
        )
        
        # Estatísticas do target
        n_positives = df[target_col].sum()
        pct_positives = n_positives / len(df) * 100
        logger.info(f"  {target_col}: {n_positives} positivos ({pct_positives:.1f}%)")
    
    # Remover colunas auxiliares
    df = df.drop(columns=['proxima_data', 'dias_ate_proxima'])
    
    logger.info("✓ Targets multi-horizonte criados")
    
    return df


def create_temporal_splits(
    df: pd.DataFrame,
    frac_train: float = 0.6,
    frac_val: float = 0.2,
    date_column: str = 'data'
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Cria splits temporais (treino/validação/teste) respeitando ordenação.
    
    CRÍTICO: Split temporal previne vazamento de dados (data leakage).
    Nunca usar shuffle=True em séries temporais!
    
    Args:
        df: DataFrame com coluna de data
        frac_train: Fração para treino (padrão: 60%)
        frac_val: Fração para validação (padrão: 20%)
        date_column: Nome da coluna de data
        
    Returns:
        Tuple (mask_train, mask_val, mask_test) - Máscaras booleanas
        
    Example:
        >>> mask_train, mask_val, mask_test = create_temporal_splits(df)
        >>> X_train = df.loc[mask_train, features]
        >>> y_train = df.loc[mask_train, 'target']
    """
    logger.info(f"Criando splits temporais: {frac_train:.0%} treino, {frac_val:.0%} val, {1-frac_train-frac_val:.0%} teste")
    
    df_ord = df.sort_values(date_column).reset_index(drop=True)
    
    # Quantis temporais
    q1 = df_ord[date_column].quantile(frac_train)
    q2 = df_ord[date_column].quantile(frac_train + frac_val)
    
    # Máscaras booleanas (aplicadas ao df original, não ordenado)
    mask_train = df[date_column] <= q1
    mask_val = (df[date_column] > q1) & (df[date_column] <= q2)
    mask_test = df[date_column] > q2
    
    # Estatísticas
    logger.info(f"  Treino: {mask_train.sum()} registros ({mask_train.sum()/len(df)*100:.1f}%)")
    logger.info(f"  Validação: {mask_val.sum()} registros ({mask_val.sum()/len(df)*100:.1f}%)")
    logger.info(f"  Teste: {mask_test.sum()} registros ({mask_test.sum()/len(df)*100:.1f}%)")
    
    # Verificar datas
    logger.info(f"  Datas treino: {df.loc[mask_train, date_column].min()} a {df.loc[mask_train, date_column].max()}")
    logger.info(f"  Datas val: {df.loc[mask_val, date_column].min()} a {df.loc[mask_val, date_column].max()}")
    logger.info(f"  Datas teste: {df.loc[mask_test, date_column].min()} a {df.loc[mask_test, date_column].max()}")
    
    return mask_train, mask_val, mask_test


def balance_classes(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'smote',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balanceia classes usando SMOTE ou undersampling.
    
    ATENÇÃO: Use com cuidado. SMOTE pode causar overfitting em alguns casos.
    Recomenda-se usar apenas em treino, nunca em validação/teste.
    
    Args:
        X: Features
        y: Target
        method: Método ('smote', 'random_undersample', 'none')
        random_state: Seed para reprodutibilidade
        
    Returns:
        Tuple (X_balanced, y_balanced)
    """
    if method == 'none':
        return X, y
    
    logger.info(f"Balanceando classes usando {method}...")
    
    initial_counts = y.value_counts().to_dict()
    logger.info(f"  Distribuição inicial: {initial_counts}")
    
    if method == 'smote':
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=random_state)
    
    elif method == 'random_undersample':
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=random_state)
    
    else:
        raise ValueError(f"Método desconhecido: {method}")
    
    X_balanced, y_balanced = sampler.fit_resample(X, y)
    
    final_counts = pd.Series(y_balanced).value_counts().to_dict()
    logger.info(f"  Distribuição final: {final_counts}")
    
    return X_balanced, y_balanced


def create_stratified_folds(
    df: pd.DataFrame,
    target_col: str,
    n_folds: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Cria folds estratificados para validação cruzada.
    
    Args:
        df: DataFrame completo
        target_col: Nome da coluna target
        n_folds: Número de folds
        random_state: Seed
        
    Returns:
        Lista de tuples (train_indices, val_indices)
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    folds = []
    for train_idx, val_idx in skf.split(X, y):
        folds.append((train_idx, val_idx))
    
    logger.info(f"Criados {n_folds} folds estratificados")
    
    return folds


def calculate_class_weights(y: pd.Series) -> dict:
    """
    Calcula pesos de classe para lidar com desbalanceamento.
    
    Args:
        y: Target series
        
    Returns:
        Dict {classe: peso}
        
    Example:
        >>> weights = calculate_class_weights(y_train)
        >>> # Usar em RandomForest: class_weight=weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    class_weights = dict(zip(classes, weights))
    
    logger.info(f"Pesos de classe calculados: {class_weights}")
    
    return class_weights
