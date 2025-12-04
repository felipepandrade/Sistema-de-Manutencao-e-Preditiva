"""
src/inference/predictor.py
Pipeline de inferência: carrega modelos e gera predições.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json
import joblib

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PredictorPipeline:
    """
    Pipeline de inferência: carrega modelos versionados e gera predições.
    """
    
    def __init__(self, model_version: str = 'latest', config: Dict = None):
        """
        Args:
            model_version: Versão do modelo (ex: 'latest', 'v20251203_142530')
            config: Configuração (carregada do YAML)
        """
        self.model_version = model_version
        self.config = config
        self.models = {}
        self.metadata = {}
        
        if config:
            try:
                self._load_models()
            except Exception as e:
                logger.error(f"Erro ao carregar modelos na inicialização: {e}")
                # Não propagar o erro, mas logar para o usuário saber o que aconteceu
                # self.models ficará vazio, e o predict() retornará erro mais claro
    
    def _load_models(self):
        """Carrega modelos e metadados da versão especificada."""
        models_dir = Path(self.config['paths']['models'])
        version_path = models_dir / self.model_version
        
        if not version_path.exists():
            raise FileNotFoundError(f"Versão não encontrada: {version_path}")
        
        # Carregar metadados
        metadata_file = version_path / 'metadata.json'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadados não encontrados: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Carregar modelos por horizonte
        for h_str in self.metadata['models_by_horizon'].keys():
            h = int(h_str)
            
            model_path = version_path / f'modelo_{h}d.pkl'
            calibrador_path = version_path / f'calibrador_{h}d.pkl'
            
            if not model_path.exists():
                logger.warning(f"Modelo {h}d não encontrado, pulando...")
                continue
            
            self.models[h] = {
                'model': joblib.load(model_path),
                'calibrador': joblib.load(calibrador_path) if calibrador_path.exists() else None,
                'threshold': self.metadata['models_by_horizon'][h_str]['best_threshold']
            }
        
        logger.info(f"Modelos carregados: versão {self.model_version}, {len(self.models)} horizontes")
    
    def predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Gera predições multi-horizonte para todos os ativos.
        
        Args:
            df_features: DataFrame com features (saída do FeatureEngineeringPipeline)
            
        Returns:
            DataFrame com predições por ativo e horizonte
        """
        if not self.models:
            error_msg = (
                "Nenhum modelo carregado. Possíveis causas:\n"
                "1. Nenhum modelo foi treinado ainda - Execute o treinamento primeiro\n"
                "2. Modelos não foram encontrados no diretório especificado\n"
                "3. Erro ao carregar modelos - Verifique os logs acima"
            )
            raise ValueError(error_msg)
        
        feature_names = self.metadata['features']
        ativos_unicos = df_features['ativo_unico'].unique()
        
        predictions_list = []
        
        for ativo in ativos_unicos:
            df_ativo = df_features[df_features['ativo_unico'] == ativo]
            
            if len(df_ativo) == 0:
                continue
            
            # Pegar último registro (estado atual)
            ultimo_registro = df_ativo.iloc[-1]
            
            # Verificar se todas as features estão presentes
            missing_features = set(feature_names) - set(df_ativo.columns)
            if missing_features:
                logger.warning(f"Features faltando para {ativo}: {missing_features}")
                continue
            
            X_pred = ultimo_registro[feature_names].values.reshape(1, -1)
            X_pred_df = pd.DataFrame(X_pred, columns=feature_names)
            
            predicao_ativo = {'ativo_unico': ativo}
            
            # Predições para cada horizonte
            for h in sorted(self.models.keys()):
                model_info = self.models[h]
                model = model_info['model']
                calibrador = model_info['calibrador']
                threshold = model_info['threshold']
                
                try:
                    # Probabilidade calibrada
                    if calibrador:
                        prob = calibrador.predict_proba(X_pred_df)[0, 1]
                    else:
                        prob = model.predict_proba(X_pred_df)[0, 1]
                    
                    prob = float(prob)
                    
                    # Classificação de risco
                    classe = self._classify_risk(prob)
                    
                    # Intervalo de confiança (simplificado)
                    ic_lower = max(0, prob - 0.05)
                    ic_upper = min(1, prob + 0.05)
                    
                    predicao_ativo[f'ProbML_Media_{h}d'] = prob
                    predicao_ativo[f'ICML_{h}d'] = f"{ic_lower:.2f} - {ic_upper:.2f}"
                    predicao_ativo[f'Classe_{h}d'] = classe
                    
                    # Guardar nome do modelo para o horizonte 30d
                    if h == 30:
                        predicao_ativo['ModeloEst'] = self.metadata['models_by_horizon'][str(h)]['model_name']
                
                except Exception as e:
                    logger.warning(f"Erro na predição para {ativo}, horizonte {h}d: {e}")
                    predicao_ativo[f'ProbML_Media_{h}d'] = np.nan
                    predicao_ativo[f'ICML_{h}d'] = 'N/A'
                    predicao_ativo[f'Classe_{h}d'] = 'Erro'
            
            predictions_list.append(predicao_ativo)
        
        df_predictions = pd.DataFrame(predictions_list)
        
        logger.info(f"Predições geradas para {len(df_predictions)} ativos")
        
        return df_predictions
    
    def _classify_risk(self, prob: float) -> str:
        """Classifica risco baseado em thresholds configurados."""
        if not self.config:
            # Thresholds padrão
            if prob >= 0.70:
                return "Alto Risco"
            elif prob >= 0.30:
                return "Médio Risco"
            else:
                return "Baixo Risco"
        
        thresholds = self.config['inference']['risk_thresholds']
        
        if prob >= thresholds['alto']:
            return "Alto Risco"
        elif prob >= thresholds['medio']:
            return "Médio Risco"
        else:
            return "Baixo Risco"
