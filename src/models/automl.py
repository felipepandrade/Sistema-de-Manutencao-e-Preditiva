"""
src/models/automl.py
Módulo de AutoML Híbrido (H2O, FLAML ou Manual).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import logging
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Tentar importar bibliotecas opcionais
try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False

try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False


class AutoMLHybrid:
    """
    Wrapper híbrido para AutoML.
    Seleciona automaticamente o melhor engine baseado nos recursos disponíveis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = self._detect_best_engine()
        self.model = None
        self.best_params = {}
        
    def _detect_best_engine(self) -> str:
        """Detecta qual engine usar baseado em memória e disponibilidade."""
        # Verificar configuração forçada
        forced_engine = self.config.get('models', {}).get('automl', {}).get('engine', 'auto')
        if forced_engine != 'auto':
            return forced_engine
            
        # Verificar memória disponível (GB)
        mem_gb = psutil.virtual_memory().available / (1024**3)
        
        if mem_gb >= 3.0 and H2O_AVAILABLE:
            logger.info(f"AutoML: H2O selecionado (Memória: {mem_gb:.1f}GB)")
            return "H2O"
        elif mem_gb >= 1.0 and FLAML_AVAILABLE:
            logger.info(f"AutoML: FLAML selecionado (Memória: {mem_gb:.1f}GB)")
            return "FLAML"
        else:
            logger.info(f"AutoML: Manual Pipeline selecionado (Memória: {mem_gb:.1f}GB)")
            return "Manual"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        time_limit_mins: int = 5
    ):
        """Treina o modelo usando o engine selecionado."""
        logger.info(f"Iniciando treinamento AutoML com engine: {self.engine}")
        
        if self.engine == "H2O":
            self._fit_h2o(X_train, y_train, X_val, y_val, time_limit_mins)
        elif self.engine == "FLAML":
            self._fit_flaml(X_train, y_train, time_limit_mins)
        else:
            self._fit_manual(X_train, y_train)
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Gera predições (classes)."""
        if self.engine == "H2O":
            hf = h2o.H2OFrame(X)
            preds = self.model.predict(hf)
            return preds['predict'].as_data_frame().values.flatten()
        elif self.engine == "FLAML":
            return self.model.predict(X)
        else:
            return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Gera probabilidades."""
        if self.engine == "H2O":
            hf = h2o.H2OFrame(X)
            preds = self.model.predict(hf)
            # H2O retorna [predict, p0, p1, ...]
            return preds.as_data_frame().iloc[:, 1:].values
        elif self.engine == "FLAML":
            return self.model.predict_proba(X)
        else:
            return self.model.predict_proba(X)

    def _fit_h2o(self, X_train, y_train, X_val, y_val, time_limit_mins):
        """Implementação H2O."""
        try:
            h2o.init(max_mem_size="2G", verbose=False)
            
            # Combinar X e y para H2OFrame
            train_df = X_train.copy()
            train_df['target'] = y_train
            hf_train = h2o.H2OFrame(train_df)
            hf_train['target'] = hf_train['target'].asfactor()
            
            if X_val is not None:
                val_df = X_val.copy()
                val_df['target'] = y_val
                hf_val = h2o.H2OFrame(val_df)
                hf_val['target'] = hf_val['target'].asfactor()
            else:
                hf_val = None
            
            self.model = H2OAutoML(
                max_runtime_secs=time_limit_mins * 60,
                seed=42,
                verbosity='info'
            )
            
            self.model.train(y='target', training_frame=hf_train, validation_frame=hf_val)
            logger.info(f"H2O Leader: {self.model.leader.model_id}")
            
        except Exception as e:
            logger.error(f"Erro no H2O: {e}. Fallback para Manual.")
            self.engine = "Manual"
            self._fit_manual(X_train, y_train)

    def _fit_flaml(self, X_train, y_train, time_limit_mins):
        """Implementação FLAML."""
        try:
            self.model = AutoML()
            self.model.fit(
                X_train, y_train,
                task="classification",
                time_budget=time_limit_mins * 60,
                verbose=0
            )
            logger.info(f"FLAML Best Config: {self.model.best_config}")
            
        except Exception as e:
            logger.error(f"Erro no FLAML: {e}. Fallback para Manual.")
            self.engine = "Manual"
            self._fit_manual(X_train, y_train)

    def _fit_manual(self, X_train, y_train):
        """Implementação Manual (RandomForest com GridSearch)."""
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        logger.info(f"Manual Best Params: {self.best_params}")
