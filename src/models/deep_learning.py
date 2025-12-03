"""
src/models/deep_learning.py
Módulo de Deep Learning para séries temporais (LSTM/GRU).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Tentar importar tensorflow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow não encontrado. Deep Learning indisponível.")


class TimeSeriesDL(BaseEstimator, ClassifierMixin):
    """
    Classificador Deep Learning para séries temporais (LSTM/GRU).
    Compatível com interface scikit-learn.
    """
    
    def __init__(
        self,
        architecture: str = 'lstm',
        timesteps: int = 5,
        epochs: int = 30,
        batch_size: int = 32,
        units: int = 50,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        verbose: int = 0
    ):
        self.architecture = architecture
        self.timesteps = timesteps
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple] = None):
        """Treina o modelo."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow necessário para treinar TimeSeriesDL")
            
        # Escalar dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Preparar sequências (sliding window)
        X_seq, y_seq = self._create_sequences(X_scaled, y.values, self.timesteps)
        
        if len(X_seq) == 0:
            logger.warning("Dados insuficientes para criar sequências com timesteps={self.timesteps}")
            return self
            
        # Construir modelo
        self.model = self._build_model(input_shape=(self.timesteps, X.shape[1]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        # Dados de validação
        val_seq = None
        if validation_data:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val.values, self.timesteps)
            if len(X_val_seq) > 0:
                val_seq = (X_val_seq, y_val_seq)
        
        # Treinar
        self.history = self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=val_seq,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prediz classes."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Prediz probabilidades."""
        if not TF_AVAILABLE or self.model is None:
            return np.zeros((len(X), 2))
            
        X_scaled = self.scaler.transform(X)
        
        # Para predição, precisamos de sequências.
        # Estratégia: Padding para os primeiros registros que não têm histórico suficiente
        # Ou: Simplificação - usar apenas o último timestep repetido (não ideal, mas funcional para compatibilidade de shape)
        
        # Abordagem correta: Recriar sequências. Mas isso reduz o tamanho do output.
        # Para manter compatibilidade 1:1 com X, vamos fazer padding.
        
        X_seq = []
        for i in range(len(X_scaled)):
            if i < self.timesteps - 1:
                # Padding com zeros ou repetindo o primeiro
                pad_len = self.timesteps - 1 - i
                seq = np.vstack([X_scaled[:i+1]] * (pad_len + 1))[:self.timesteps]
            else:
                seq = X_scaled[i - self.timesteps + 1 : i + 1]
            X_seq.append(seq)
            
        X_seq = np.array(X_seq)
        
        # Predizer
        preds = self.model.predict(X_seq, verbose=0)
        
        # Formatar para [prob_0, prob_1]
        return np.hstack([1 - preds, preds])

    def _build_model(self, input_shape):
        """Constrói arquitetura Keras."""
        model = Sequential()
        
        if self.architecture == 'lstm':
            model.add(LSTM(self.units, input_shape=input_shape, return_sequences=False))
        elif self.architecture == 'gru':
            model.add(GRU(self.units, input_shape=input_shape, return_sequences=False))
        elif self.architecture == 'bilstm':
            model.add(Bidirectional(LSTM(self.units, return_sequences=False), input_shape=input_shape))
        else:
            raise ValueError(f"Arquitetura desconhecida: {self.architecture}")
            
        model.add(Dropout(self.dropout))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def _create_sequences(self, X, y, time_steps=1):
        """Cria sequências para séries temporais."""
        Xs, ys = [], []
        for i in range(len(X) - time_steps + 1):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps - 1])
        return np.array(Xs), np.array(ys)
