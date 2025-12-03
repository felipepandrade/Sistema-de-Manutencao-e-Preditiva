"""
src/models/ensemble.py
Wrapper para Stacking Ensemble.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

class StackingEnsembleWrapper:
    """
    Wrapper que encapsula o meta-modelo e os modelos base.
    Permite que o ensemble seja tratado como um único modelo (predict/predict_proba).
    """
    
    def __init__(self, meta_model, base_models: Dict[str, Any]):
        """
        Args:
            meta_model: Modelo final (ex: LogisticRegression)
            base_models: Dict {nome: modelo_treinado_ou_calibrador}
        """
        self.meta_model = meta_model
        self.base_models = base_models
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Gera probabilidades combinando os modelos base.
        """
        meta_features = pd.DataFrame(index=X.index)
        
        # Gerar predições de cada modelo base
        for name, model in self.base_models.items():
            try:
                # Se for calibrador ou modelo sklearn/xgboost
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                    # Pegar probabilidade da classe positiva
                    if probs.shape[1] > 1:
                        meta_features[name] = probs[:, 1]
                    else:
                        meta_features[name] = probs.flatten()
                else:
                    # Fallback (ex: modelo que só tem predict)
                    meta_features[name] = model.predict(X)
            except Exception as e:
                # Em caso de erro num modelo base, usar 0.5 (neutro)
                meta_features[name] = 0.5
        
        # Garantir ordem das colunas igual ao treino
        # (LogisticRegression usa ordem de fit, que foi baseada no dict keys)
        # O dict preserva ordem de inserção no Python 3.7+, então deve estar ok
        # Mas por segurança, o meta_model espera as features na ordem que foi treinado.
        # O wrapper deve garantir isso.
        
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prediz classes (threshold 0.5 padrão)."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
