"""
src/models/classical.py
Modelos clássicos de Machine Learning com calibração.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb

from src.utils.logging_config import get_logger
from src.utils.metrics import calculate_classification_metrics

logger = get_logger(__name__)


def montar_modelos_base() -> Dict[str, Any]:
    """
    Factory de modelos clássicos.
    
    Returns:
        Dict {nome: modelo}
    """
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            max_depth=6
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100,
            random_state=42,
            verbose=-1,
            max_depth=6
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10
        )
    }
    
    # Adicionar CatBoost se disponível
    try:
        from catboost import CatBoostClassifier
        models['CatBoost'] = CatBoostClassifier(
            iterations=100,
            random_state=42,
            verbose=0,
            depth=6
        )
    except ImportError:
        logger.info("CatBoost não disponível")
    
    return models


def calibrate_model(model, X_val, y_val, method: str = 'sigmoid'):
    """
    Calibra probabilidades do modelo.
    
    Args:
        model: Modelo treinado
        X_val: Features de validação
        y_val: Target de validação
        method: Método de calibração ('sigmoid' ou 'isotonic')
        
    Returns:
        Modelo calibrado
    """
    logger.info(f"Calibrando modelo com método {method}...")
    
    calibrador = CalibratedClassifierCV(model, method=method, cv='prefit')
    calibrador.fit(X_val, y_val)
    
    return calibrador


def optimize_threshold(
    calibrador,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = 'f1'
) -> float:
    """
    Otimiza threshold de classificação.
    
    Args:
        calibrador: Modelo calibrado
        X_val: Features de validação
        y_val: Target de validação
        metric: Métrica para otimizar ('f1', 'precision', 'recall')
        
    Returns:
        Threshold ótimo
    """
    from sklearn.metrics import precision_score, recall_score
    
    y_prob = calibrador.predict_proba(X_val)[:, 1]
    
    best_thr = 0.5
    best_score = -1
    
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= thr).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_val, y_pred, zero_division=0) if len(np.unique(y_val)) > 1 else 0.0
        elif metric == 'precision':
            score = precision_score(y_val, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_val, y_pred, zero_division=0)
        else:
            raise ValueError(f"Métrica '{metric}' não suportada")
        
        if score > best_score:
            best_score = score
            best_thr = thr
    
    logger.info(f"Threshold ótimo: {best_thr:.3f} (Score={best_score:.3f})")
    
    return best_thr


def train_classical_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_names: List[str],
    calibration_method: str = 'sigmoid'
) -> Dict[str, Dict]:
    """
    Treina múltiplos modelos clássicos com calibração.
    
    Args:
        X_train, y_train: Dados de treino
        X_val, y_val: Dados de validação
        X_test, y_test: Dados de teste
        model_names: Lista de nomes de modelos a treinar
        calibration_method: Método de calibração
        
    Returns:
        Dict {nome_modelo: {model, calibrador, metrics, ...}}
    """
    models_base = montar_modelos_base()
    results = {}
    
    for name in model_names:
        if name not in models_base:
            logger.warning(f"Modelo {name} não disponível, pulando...")
            continue
        
        try:
            logger.info(f"Treinando {name}...")
            model = models_base[name]
            
            # Treinar
            model.fit(X_train, y_train)
            
            # Calibrar
            calibrador = calibrate_model(model, X_val, y_val, method=calibration_method)
            
            # Otimizar threshold
            best_thr = optimize_threshold(calibrador, X_val, y_val, metric='f1')
            
            # Métricas em validação
            y_prob_val = calibrador.predict_proba(X_val)[:, 1]
            y_pred_val = (y_prob_val >= best_thr).astype(int)
            metrics_val = calculate_classification_metrics(y_val, y_pred_val, y_prob_val, best_thr)
            
            # Métricas em teste
            y_prob_test = calibrador.predict_proba(X_test)[:, 1]
            y_pred_test = (y_prob_test >= best_thr).astype(int)
            metrics_test = calculate_classification_metrics(y_test, y_pred_test, y_prob_test, best_thr)
            
            results[name] = {
                'model': model,
                'calibrador': calibrador,
                'best_threshold': float(best_thr),
                'metrics_val': metrics_val,
                'metrics_test': metrics_test
            }
            
            logger.info(f"{name} treinado: F1-VAL={metrics_val['F1-Score']:.3f}, F1-TEST={metrics_test['F1-Score']:.3f}")
            
        except Exception as e:
            logger.error(f"Erro ao treinar {name}: {e}", exc_info=True)
    
    return results
