"""
src/models/reliability.py
Módulo de Análise de Confiabilidade (Survival Analysis).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Tentar importar lifelines
try:
    from lifelines import KaplanMeierFitter, WeibullFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logger.warning("Lifelines não encontrado. Análise de confiabilidade indisponível.")


class ReliabilityAnalyzer:
    """
    Analisador de confiabilidade usando Survival Analysis.
    Estima curvas de sobrevivência e risco para ativos.
    """
    
    def __init__(self):
        self.kmf = None
        self.wf = None
        
    def fit(self, df: pd.DataFrame, time_col: str = 'tbf', event_col: str = 'event_occurred'):
        """
        Ajusta modelos de sobrevivência (Kaplan-Meier e Weibull).
        
        Args:
            df: DataFrame com dados históricos
            time_col: Coluna com tempo até evento (TBF)
            event_col: Coluna indicando se evento ocorreu (1) ou censura (0)
        """
        if not LIFELINES_AVAILABLE:
            return
            
        # Preparar dados
        # Se não tiver coluna de evento explícita, assumimos que todos são falhas (1)
        # a menos que seja o último registro de cada ativo (censura)
        
        data = df.copy()
        if event_col not in data.columns:
            # Simplificação: assumir que todos TBFs registrados são falhas
            # Para análise mais robusta, precisaríamos identificar censura (ativo ainda rodando)
            data[event_col] = 1
            
        T = data[time_col]
        E = data[event_col]
        
        try:
            # Kaplan-Meier (Não-paramétrico)
            self.kmf = KaplanMeierFitter()
            self.kmf.fit(T, event_observed=E, label='Kaplan-Meier')
            
            # Weibull (Paramétrico)
            self.wf = WeibullFitter()
            self.wf.fit(T, event_observed=E, label='Weibull')
            
            logger.info(f"Modelos de confiabilidade ajustados. Weibull rho: {self.wf.rho_:.2f}")
            
        except Exception as e:
            logger.error(f"Erro ao ajustar modelos de confiabilidade: {e}")

    def predict_survival_probability(self, t: float) -> float:
        """Prediz probabilidade de sobrevivência no tempo t."""
        if self.wf:
            return self.wf.survival_function_at_times(t).values[0]
        elif self.kmf:
            # Encontrar índice mais próximo
            idx = (np.abs(self.kmf.survival_function_.index - t)).argmin()
            return self.kmf.survival_function_.iloc[idx].values[0]
        else:
            return 0.0

    def plot_survival_curve(self) -> go.Figure:
        """Gera gráfico interativo da curva de sobrevivência."""
        if not self.kmf and not self.wf:
            return go.Figure()
            
        fig = go.Figure()
        
        if self.kmf:
            fig.add_trace(go.Scatter(
                x=self.kmf.survival_function_.index,
                y=self.kmf.survival_function_['Kaplan-Meier'],
                mode='lines',
                name='Kaplan-Meier',
                line=dict(color='blue')
            ))
            
        if self.wf:
            # Gerar pontos para Weibull
            t_max = self.kmf.survival_function_.index.max() if self.kmf else 100
            t_range = np.linspace(0, t_max, 100)
            s_weibull = self.wf.survival_function_at_times(t_range)
            
            fig.add_trace(go.Scatter(
                x=t_range,
                y=s_weibull,
                mode='lines',
                name='Weibull Fit',
                line=dict(color='red', dash='dash')
            ))
            
        fig.update_layout(
            title="Curva de Sobrevivência (Confiabilidade)",
            xaxis_title="Tempo (dias)",
            yaxis_title="Probabilidade de Sobrevivência",
            yaxis=dict(range=[0, 1]),
            template="plotly_white"
        )
        
        return fig

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo estatístico."""
        if not self.wf:
            return {}
            
        return {
            "lambda_": self.wf.lambda_,
            "rho_": self.wf.rho_,
            "median_survival_time": self.wf.median_survival_time_,
            "mean_survival_time": self.wf.mean_survival_time_
        }
