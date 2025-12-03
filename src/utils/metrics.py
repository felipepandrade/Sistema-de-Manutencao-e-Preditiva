"""
src/utils/metrics.py
Métricas customizadas para avaliação de modelos preditivos.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calcula métricas de classificação completas.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições binárias
        y_prob: Probabilidades (opcional, para AUC)
        threshold: Threshold usado para classificação
        
    Returns:
        Dict com métricas calculadas
    """
    metrics = {}
    
    # Proteção contra arrays vazios
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Arrays vazios fornecidos para cálculo de métricas")
        return {
            'Accuracy': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1-Score': 0.0,
            'AUC-ROC': 0.0
        }
    
    # Métricas básicas
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1-Score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC (se probabilidades fornecidas)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
            metrics['AUC-PR'] = average_precision_score(y_true, y_prob)
        except ValueError as e:
            logger.warning(f"Erro ao calcular AUC: {e}")
            metrics['AUC-ROC'] = 0.0
            metrics['AUC-PR'] = 0.0
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['True Negatives'] = int(tn)
        metrics['False Positives'] = int(fp)
        metrics['False Negatives'] = int(fn)
        metrics['True Positives'] = int(tp)
        
        # Especificidade
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Threshold usado
    metrics['Threshold'] = float(threshold)
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
) -> None:
    """
    Imprime relatório de classificação detalhado.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        target_names: Nomes das classes (padrão: ['Negativo', 'Positivo'])
    """
    if target_names is None:
        target_names = ['Negativo', 'Positivo']
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\n" + "="*60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*60)
    print(report)
    print("="*60 + "\n")


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 100.0,  # Custo de falso positivo (manutenção desnecessária)
    cost_fn: float = 1000.0,  # Custo de falso negativo (falha não detectada)
    cost_tp: float = 50.0,  # Custo de manutenção preventiva
    revenue_tn: float = 0.0  # Benefício de evitar manutenção corretamente
) -> Dict[str, float]:
    """
    Calcula métricas de negócio baseadas em custos.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        cost_fp: Custo de falso positivo
        cost_fn: Custo de falso negativo
        cost_tp: Custo de verdadeiro positivo
        revenue_tn: Receita de verdadeiro negativo
        
    Returns:
        Dict com métricas de negócio
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        logger.warning("Matriz de confusão inválida para cálculo de métricas de negócio")
        return {}
    
    tn, fp, fn, tp = cm.ravel()
    
    # Custos totais
    total_cost_fp = fp * cost_fp
    total_cost_fn = fn * cost_fn
    total_cost_tp = tp * cost_tp
    total_revenue_tn = tn * revenue_tn
    
    # Custo/lucro total
    total_cost = total_cost_fp + total_cost_fn + total_cost_tp
    total_revenue = total_revenue_tn
    net_result = total_revenue - total_cost
    
    # ROI (Return on Investment)
    roi = (total_revenue - total_cost) / total_cost * 100 if total_cost > 0 else 0.0
    
    return {
        'Total Cost FP': total_cost_fp,
        'Total Cost FN': total_cost_fn,
        'Total Cost TP': total_cost_tp,
        'Total Revenue TN': total_revenue_tn,
        'Total Cost': total_cost,
        'Total Revenue': total_revenue,
        'Net Result': net_result,
        'ROI (%)': roi
    }


def compare_models_metrics(
    results_dict: Dict[str, Dict[str, float]],
    sort_by: str = 'F1-Score'
) -> pd.DataFrame:
    """
    Compara métricas de múltiplos modelos.
    
    Args:
        results_dict: Dict {nome_modelo: {metricas}}
        sort_by: Métrica para ordenar (padrão: F1-Score)
        
    Returns:
        DataFrame com comparação ordenada
    """
    df = pd.DataFrame(results_dict).T
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    return df


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    Formata métricas como tabela ASCII.
    
    Args:
        metrics: Dict de métricas
        
    Returns:
        String formatada como tabela
    """
    lines = []
    lines.append("┌" + "─" * 30 + "┬" + "─" * 12 + "┐")
    lines.append("│ " + "Métrica".ljust(28) + " │ " + "Valor".rjust(10) + " │")
    lines.append("├" + "─" * 30 + "┼" + "─" * 12 + "┤")
    
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            value_str = f"{value:,d}"
        elif isinstance(value, (float, np.floating)):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        
        lines.append("│ " + key.ljust(28) + " │ " + value_str.rjust(10) + " │")
    
    lines.append("└" + "─" * 30 + "┴" + "─" * 12 + "┘")
    
    return "\n".join(lines)
