"""
src/explainability/ig_explainer.py
Módulo de explicabilidade usando Integrated Gradients para Deep Learning.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Optional, Any
import logging
import plotly.graph_objects as go

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class IGExplainer:
    """
    Implementação de Integrated Gradients para explicar modelos Keras/TensorFlow.
    """
    
    def __init__(self, model: Any, baseline: Optional[np.ndarray] = None):
        """
        Args:
            model: Modelo Keras treinado
            baseline: Baseline input (geralmente zeros). Se None, usa zeros.
        """
        self.model = model
        self.baseline = baseline
        
    def explain(
        self,
        X_sample: np.ndarray,
        feature_names: List[str],
        m_steps: int = 50
    ) -> pd.DataFrame:
        """
        Calcula atribuição de features usando Integrated Gradients.
        
        Args:
            X_sample: Amostra de entrada (numpy array)
            feature_names: Nomes das features
            m_steps: Número de passos de interpolação
            
        Returns:
            DataFrame com importância das features
        """
        if self.baseline is None:
            self.baseline = np.zeros_like(X_sample)
            
        # Converter para tensores
        X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)
        baseline_tensor = tf.convert_to_tensor(self.baseline, dtype=tf.float32)
        
        # Interpolação
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
        
        # Gerar inputs interpolados
        # Shape: (m_steps+1, timesteps, features) ou (m_steps+1, features)
        # Precisamos lidar com a dimensão extra se for sequência
        
        # Assumindo input (timesteps, features) para um único sample
        # Precisamos expandir alphas para fazer broadcast correto
        
        # Caso 1: Input tabular (features,)
        if len(X_sample.shape) == 1:
            alphas_x = alphas[:, tf.newaxis]
        # Caso 2: Input sequencial (timesteps, features)
        elif len(X_sample.shape) == 2:
            alphas_x = alphas[:, tf.newaxis, tf.newaxis]
        # Caso 3: Batch (batch, timesteps, features) - não suportado diretamente aqui para simplificação
        else:
            raise ValueError(f"Shape não suportado: {X_sample.shape}")
            
        delta = X_tensor - baseline_tensor
        interpolated_inputs = baseline_tensor + alphas_x * delta
        
        # Calcular gradientes
        with tf.GradientTape() as tape:
            tape.watch(interpolated_inputs)
            predictions = self.model(interpolated_inputs)
            
        grads = tape.gradient(predictions, interpolated_inputs)
        
        # Aproximação integral (Regra do Trapézio ou Riemann)
        avg_grads = tf.reduce_mean(grads, axis=0)
        
        # Integrated Gradients = (Input - Baseline) * AvgGrads
        integrated_gradients = delta * avg_grads
        
        # Se for sequencial, somar ao longo do tempo para ter importância por feature
        if len(integrated_gradients.shape) == 2:
            feature_importance = tf.reduce_sum(integrated_gradients, axis=0)
        else:
            feature_importance = integrated_gradients
            
        # Converter para numpy e criar DataFrame
        importances = feature_importance.numpy()
        
        df_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Ordenar por valor absoluto
        df_imp['abs_importance'] = df_imp['importance'].abs()
        df_imp = df_imp.sort_values('abs_importance', ascending=False)
        
        return df_imp

    def plot_feature_importance(self, df_imp: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """Plota importância das features."""
        df_plot = df_imp.head(top_n)
        
        fig = go.Figure(go.Bar(
            x=df_plot['importance'],
            y=df_plot['feature'],
            orientation='h',
            marker=dict(
                color=df_plot['importance'],
                colorscale='RdBu',
                midpoint=0
            )
        ))
        
        fig.update_layout(
            title="Importância de Features (Integrated Gradients)",
            xaxis_title="Atribuição",
            yaxis_title="Feature",
            yaxis=dict(autorange="reversed"),
            height=500
        )
        
        return fig
