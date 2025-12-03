"""
src/maintenance/performance_tracker.py
Módulo de rastreamento de performance e loop de feedback.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.metrics import calculate_classification_metrics

logger = get_logger(__name__)


class PerformanceTracker:
    """
    Rastreia a performance dos modelos em produção comparando predições com falhas reais.
    """
    
    def __init__(self, logs_dir: Path = Path("outputs/logs")):
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.logs_dir / "prediction_history.json"
        self._load_history()

    def _load_history(self):
        """Carrega histórico de predições."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar histórico: {e}")
                self.history = []
        else:
            self.history = []

    def log_prediction(
        self,
        ativo_unico: str,
        data_predicao: str,
        horizonte: int,
        probabilidade: float,
        classe_predita: str,
        modelo_versao: str
    ):
        """Registra uma predição realizada."""
        entry = {
            "ativo_unico": ativo_unico,
            "data_predicao": data_predicao,
            "horizonte": horizonte,
            "probabilidade": probabilidade,
            "classe_predita": classe_predita,
            "modelo_versao": modelo_versao,
            "resultado_real": None,  # A ser preenchido depois
            "data_resultado": None
        }
        self.history.append(entry)
        self._save_history()

    def register_outcome(
        self,
        ativo_unico: str,
        data_evento: str,
        houve_falha: bool
    ):
        """
        Registra o resultado real (feedback) para predições passadas.
        Associa falhas às predições que estavam dentro do horizonte.
        """
        data_evento_dt = pd.to_datetime(data_evento)
        updated_count = 0
        
        for entry in self.history:
            if entry["ativo_unico"] == ativo_unico and entry["resultado_real"] is None:
                data_pred_dt = pd.to_datetime(entry["data_predicao"])
                horizonte_dias = entry["horizonte"]
                
                # Verificar se o evento ocorreu dentro do horizonte da predição
                delta_dias = (data_evento_dt - data_pred_dt).days
                
                if 0 <= delta_dias <= horizonte_dias:
                    entry["resultado_real"] = int(houve_falha)
                    entry["data_resultado"] = data_evento
                    updated_count += 1
        
        if updated_count > 0:
            self._save_history()
            logger.info(f"Feedback registrado para {updated_count} predições de {ativo_unico}")

    def calculate_metrics(self, horizonte: Optional[int] = None) -> Dict[str, float]:
        """Calcula métricas de performance baseadas no histórico com feedback."""
        df = pd.DataFrame(self.history)
        
        if df.empty:
            return {}
        
        # Filtrar apenas registros com resultado conhecido
        df_eval = df[df["resultado_real"].notna()].copy()
        
        if df_eval.empty:
            logger.warning("Sem dados com feedback para calcular métricas")
            return {}
        
        if horizonte:
            df_eval = df_eval[df_eval["horizonte"] == horizonte]
        
        if df_eval.empty:
            return {}
        
        y_true = df_eval["resultado_real"].astype(int).values
        # Considerar positivo se classe for Alto ou Médio Risco (simplificação)
        y_pred = df_eval["classe_predita"].apply(
            lambda x: 1 if x in ["Alto Risco", "Médio Risco"] else 0
        ).values
        
        metrics = calculate_classification_metrics(y_true, y_pred)
        return metrics

    def _save_history(self):
        """Salva histórico em disco."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_metrics_over_time(
        self,
        horizonte: Optional[int] = None,
        window_days: int = 30
    ) -> pd.DataFrame:
        """
        Retorna métricas de performance ao longo do tempo.
        
        Args:
            horizonte: Horizonte específico (None = todos)
            window_days: Tamanho da janela para calcular métricas
            
        Returns:
            DataFrame com métricas temporais
        """
        df = pd.DataFrame(self.history)
        
        if df.empty:
            return pd.DataFrame()
        
        # Filtrar apenas com feedback
        df_eval = df[df["resultado_real"].notna()].copy()
        
        if df_eval.empty:
            return pd.DataFrame()
        
        if horizonte:
            df_eval = df_eval[df_eval["horizonte"] == horizonte]
        
        # Converter datas
        df_eval['data_predicao'] = pd.to_datetime(df_eval['data_predicao'])
        df_eval = df_eval.sort_values('data_predicao')
        
        # Calcular métricas em janelas móveis
        results = []
        
        for i in range(len(df_eval)):
            # Janela até a data atual
            window_end = df_eval.iloc[i]['data_predicao']
            window_start = window_end - pd.Timedelta(days=window_days)
            
            df_window = df_eval[
                (df_eval['data_predicao'] >= window_start) & 
                (df_eval['data_predicao'] <= window_end)
            ]
            
            if len(df_window) >= 5:  # Mínimo de amostras
                y_true = df_window['resultado_real'].astype(int).values
                y_pred = df_window['classe_predita'].apply(
                    lambda x: 1 if x in ["Alto Risco", "Médio Risco"] else 0
                ).values
                
                metrics = calculate_classification_metrics(y_true, y_pred)
                
                results.append({
                    'data': window_end,
                    'f1': metrics.get('F1-Score', 0),
                    'precision': metrics.get('Precision', 0),
                    'recall': metrics.get('Recall', 0),
                    'auc': metrics.get('AUC-ROC', 0),
                    'n_samples': len(df_window)
                })
        
        return pd.DataFrame(results)
    
    def detect_drift(
        self,
        horizonte: int = 30,
        baseline_window_days: int = 90,
        current_window_days: int = 30,
        threshold_drop: float = 0.05
    ) -> Dict[str, any]:
        """
        Detecta drift de modelo comparando performance recente vs baseline.
        
        Args:
            horizonte: Horizonte a analisar
            baseline_window_days: Janela para baseline histórico
            current_window_days: Janela para performance recente
            threshold_drop: Queda mínima para alertar
            
        Returns:
            Dict com resultados do drift detection
        """
        df = pd.DataFrame(self.history)
        
        if df.empty:
            return {"drift_detected": False, "message": "Sem dados históricos"}
        
        df_eval = df[
            (df["resultado_real"].notna()) & 
            (df["horizonte"] == horizonte)
        ].copy()
        
        if len(df_eval) < 20:
            return {"drift_detected": False, "message": "Dados insuficientes (< 20 amostras)"}
        
        df_eval['data_predicao'] = pd.to_datetime(df_eval['data_predicao'])
        df_eval = df_eval.sort_values('data_predicao')
        
        # Janela recente
        current_end = df_eval['data_predicao'].max()
        current_start = current_end - pd.Timedelta(days=current_window_days)
        df_current = df_eval[df_eval['data_predicao'] >= current_start]
        
        # Janela baseline (anterior à recente)
        baseline_end = current_start
        baseline_start = baseline_end - pd.Timedelta(days=baseline_window_days)
        df_baseline = df_eval[
            (df_eval['data_predicao'] >= baseline_start) & 
            (df_eval['data_predicao'] < baseline_end)
        ]
        
        if len(df_current) < 5 or len(df_baseline) < 5:
            return {"drift_detected": False, "message": "Amostras insuficientes nas janelas"}
        
        # Calcular métricas
        def calc_f1(df_window):
            y_true = df_window['resultado_real'].astype(int).values
            y_pred = df_window['classe_predita'].apply(
                lambda x: 1 if x in ["Alto Risco", "Médio Risco"] else 0
            ).values
            metrics = calculate_classification_metrics(y_true, y_pred)
            return metrics.get('F1-Score', 0)
        
        f1_baseline = calc_f1(df_baseline)
        f1_current = calc_f1(df_current)
        
        f1_drop = f1_baseline - f1_current
        drift_detected = f1_drop > threshold_drop
        
        return {
            "drift_detected": drift_detected,
            "f1_baseline": f1_baseline,
            "f1_current": f1_current,
            "f1_drop": f1_drop,
            "threshold": threshold_drop,
            "baseline_samples": len(df_baseline),
            "current_samples": len(df_current),
            "message": f"F1 caiu {f1_drop:.2%}" if drift_detected else "Performance estável"
        }
    
    def get_summary(self) -> Dict[str, int]:
        """Retorna sumário do tracker."""
        df = pd.DataFrame(self.history)
        
        if df.empty:
            return {
                "total_predictions": 0,
                "with_feedback": 0,
                "pending_feedback": 0
            }
        
        return {
            "total_predictions": len(df),
            "with_feedback": df["resultado_real"].notna().sum(),
            "pending_feedback": df["resultado_real"].isna().sum()
        }
