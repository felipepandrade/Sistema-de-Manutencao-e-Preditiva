"""
src/explainability/shap_explainer.py
Módulo de explicabilidade usando SHAP (SHapley Additive exPlanations).
"""

import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Dict, Any
import logging

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """
    Wrapper para explicabilidade de modelos usando SHAP.
    Focado em modelos baseados em árvore (TreeExplainer) mas compatível com outros.
    """
    
    def __init__(
        self,
        model: Any,
        X_background: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
        model_type: str = 'tree'
    ):
        """
        Inicializa o explainer.
        
        Args:
            model: Modelo treinado (sklearn, xgboost, lightgbm, etc.)
            X_background: Dados de background para estimativa (opcional para TreeExplainer)
            feature_names: Lista de nomes das features
            model_type: Tipo de modelo ('tree', 'linear', 'kernel')
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
        try:
            if model_type == 'tree':
                # TreeExplainer é otimizado para árvores
                self.explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                self.explainer = shap.LinearExplainer(model, X_background)
            else:
                # KernelExplainer é agnóstico mas lento
                self.explainer = shap.KernelExplainer(model.predict_proba, X_background)
            
            logger.info(f"SHAP Explainer inicializado ({model_type})")
            
        except Exception as e:
            logger.warning(f"Falha ao inicializar explainer específico: {e}. Tentando genérico...")
            try:
                self.explainer = shap.Explainer(model, X_background)
            except Exception as e2:
                logger.error(f"Falha crítica ao inicializar SHAP: {e2}")
                raise

    def explain(self, X: pd.DataFrame, check_additivity: bool = False) -> Any:
        """
        Calcula SHAP values para o dataset fornecido.
        
        Args:
            X: DataFrame com features
            check_additivity: Verificar aditividade (pode falhar com arredondamentos)
            
        Returns:
            Objeto shap_values
        """
        if self.explainer is None:
            raise ValueError("Explainer não inicializado")
        
        logger.info(f"Calculando SHAP values para {len(X)} amostras...")
        
        try:
            # Para TreeExplainer com modelos de classificação, shap_values pode ser uma lista (por classe)
            shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)
            
            # Se for classificação binária e retornar lista, pegar a classe positiva (índice 1)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
                
            return shap_values
            
        except Exception as e:
            logger.error(f"Erro ao calcular SHAP values: {e}")
            raise

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        max_display: int = 20,
        plot_type: str = 'dot'
    ) -> plt.Figure:
        """
        Gera plot de resumo (beeswarm ou bar).
        Retorna figura matplotlib para renderização no Streamlit.
        """
        fig = plt.figure(figsize=(10, 6 + (max_display // 5)))
        
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names if self.feature_names else X.columns,
            max_display=max_display,
            plot_type=plot_type,
            show=False
        )
        
        plt.tight_layout()
        return fig

    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        index: int = 0,
        max_display: int = 10
    ) -> plt.Figure:
        """
        Gera waterfall plot para uma única predição.
        """
        # Reconstruir objeto Explanation para waterfall plot
        # Nota: shap.plots.waterfall espera um objeto Explanation, não apenas array numpy
        
        try:
            # Tentar criar objeto Explanation manual
            explanation = shap.Explanation(
                values=shap_values[index],
                base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
                data=X.iloc[index].values,
                feature_names=self.feature_names if self.feature_names else X.columns
            )
            
            fig = plt.figure(figsize=(8, 6))
            shap.plots.waterfall(explanation, max_display=max_display, show=False)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.warning(f"Erro ao gerar waterfall plot nativo: {e}. Tentando bar plot simples.")
            
            # Fallback: Bar plot simples com matplotlib
            vals = pd.Series(shap_values[index], index=X.columns)
            vals = vals.abs().sort_values(ascending=False).head(max_display)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            vals.plot(kind='barh', ax=ax)
            ax.set_title(f"Importância de Features (Local) - Índice {index}")
            ax.invert_yaxis()
            plt.tight_layout()
            return fig

    def get_feature_importance(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Retorna DataFrame com importância global das features.
        """
        # Média absoluta dos SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        df_imp = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else X.columns,
            'importance': importance
        })
        
        df_imp = df_imp.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Normalizar para %
        df_imp['importance_pct'] = df_imp['importance'] / df_imp['importance'].sum() * 100
        
        return df_imp
