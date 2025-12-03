"""
src/models/trainer.py
Orquestrador de treinamento de modelos.
"""

import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

from src.models.classical import train_classical_models
from src.models.automl import AutoMLHybrid
from src.utils.logging_config import get_logger
from src.utils.io import save_model_artifacts
from src.utils.metrics import calculate_classification_metrics

logger = get_logger(__name__)


class ModelTrainer:
    """
    Orquestrador de treinamento: treina modelos para todos os horizontes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.horizontes = config['models']['horizontes']
        self.classical_models = config['models']['classical_models']
    
    def train_all_horizons(
        self,
        df_features: pd.DataFrame,
        feature_names: List[str],
        horizontes: List[int],
        mask_train: pd.Series,
        mask_val: pd.Series,
        mask_test: pd.Series
    ) -> Dict[int, Dict[str, Any]]:
        """
        Treina modelos para todos os horizontes e seleciona campeões.
        
        Args:
            df_features: DataFrame com features e targets
            feature_names: Lista de nomes de features
            horizontes: Lista de horizontes (dias)
            mask_train, mask_val, mask_test: Máscaras booleanas para splits
            
        Returns:
            Dict {horizonte: {model, calibrador, metrics, ...}}
        """
        results = {}
        
        for h in horizontes:
            logger.info(f"\n{'='*60}")
            logger.info(f"HORIZONTE: {h} DIAS")
            logger.info(f"{'='*60}")
            
            target_col = f'falha_em_{h}d'
            
            if target_col not in df_features.columns:
                logger.error(f"Coluna target {target_col} não encontrada!")
                continue
            
            # Preparar dados
            X_train = df_features.loc[mask_train, feature_names]
            y_train = df_features.loc[mask_train, target_col]
            
            X_val = df_features.loc[mask_val, feature_names]
            y_val = df_features.loc[mask_val, target_col]
            
            X_test = df_features.loc[mask_test, feature_names]
            y_test = df_features.loc[mask_test, target_col]
            
            logger.info(f"Treino: {len(y_train)} ({y_train.sum()} positivos, {y_train.sum()/len(y_train)*100:.1f}%)")
            logger.info(f"Validação: {len(y_val)} ({y_val.sum()} positivos, {y_val.sum()/len(y_val)*100:.1f}%)")
            logger.info(f"Teste: {len(y_test)} ({y_test.sum()} positivos, {y_test.sum()/len(y_test)*100:.1f}%)")
            
            # 1. Treinar modelos clássicos
            logger.info("Treinando modelos clássicos...")
            model_results = train_classical_models(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                model_names=self.classical_models,
                calibration_method=self.config['models']['calibration']['method']
            )
            
            # 2. Treinar AutoML (se habilitado)
            automl_engine = self.config['models']['automl'].get('engine', 'skip')
            if automl_engine != 'skip':
                logger.info(f"Treinando AutoML (Engine: {automl_engine})...")
                try:
                    automl = AutoMLHybrid(self.config)
                    automl.fit(X_train, y_train, X_val, y_val, 
                              time_limit_mins=self.config['models']['automl'].get('max_time_minutes', 5))
                    
                    # Avaliar AutoML
                    y_pred_val = automl.predict(X_val)
                    metrics_val = calculate_classification_metrics(y_val, y_pred_val)
                    
                    y_pred_test = automl.predict(X_test)
                    metrics_test = calculate_classification_metrics(y_test, y_pred_test)
                    
                    model_results['AutoML'] = {
                        'model': automl,
                        'calibrador': None,
                        'best_threshold': 0.5,
                        'metrics_val': metrics_val,
                        'metrics_test': metrics_test
                    }
                    logger.info(f"AutoML finalizado. F1-Val: {metrics_val['F1-Score']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Falha no treinamento AutoML: {e}")

            # 3. Treinar Deep Learning (se habilitado)
            dl_config = self.config['models'].get('deep_learning', {})
            if dl_config.get('enabled', False):
                logger.info("Treinando Deep Learning...")
                try:
                    from src.models.deep_learning import TimeSeriesDL
                    
                    dl_model = TimeSeriesDL(
                        architecture=dl_config.get('architecture', 'lstm'),
                        timesteps=dl_config.get('timesteps', 5),
                        epochs=dl_config.get('epochs', 30),
                        batch_size=dl_config.get('batch_size', 32)
                    )
                    
                    # Treinar
                    dl_model.fit(X_train, y_train, validation_data=(X_val, y_val))
                    
                    # Avaliar
                    y_pred_val = dl_model.predict(X_val)
                    metrics_val = calculate_classification_metrics(y_val, y_pred_val)
                    
                    y_pred_test = dl_model.predict(X_test)
                    metrics_test = calculate_classification_metrics(y_test, y_pred_test)
                    
                    model_results['DeepLearning'] = {
                        'model': dl_model,
                        'calibrador': None, # DL já retorna prob calibrada (sigmoid)
                        'best_threshold': 0.5,
                        'metrics_val': metrics_val,
                        'metrics_test': metrics_test
                    }
                    logger.info(f"Deep Learning finalizado. F1-Val: {metrics_val['F1-Score']:.3f}")
                    
                except ImportError:
                    logger.warning("TensorFlow não instalado. Pulando Deep Learning.")
                except Exception as e:
                    logger.error(f"Falha no treinamento Deep Learning: {e}")
            
            if not model_results:
                logger.error(f"Nenhum modelo treinado com sucesso para horizonte {h}d")
                continue

            # 4. Stacking Ensemble (Sinergia)
            if len(model_results) > 1:
                logger.info("Treinando Stacking Ensemble...")
                try:
                    from sklearn.linear_model import LogisticRegression
                    
                    # Coletar predições (OOF)
                    meta_X_val = pd.DataFrame()
                    meta_X_test = pd.DataFrame()
                    
                    for name, res in model_results.items():
                        # Obter probas
                        if res['calibrador']:
                            pred_val = res['calibrador'].predict_proba(X_val)[:, 1]
                            pred_test = res['calibrador'].predict_proba(X_test)[:, 1]
                        else:
                            # AutoML ou DL
                            pred_val = res['model'].predict_proba(X_val)[:, 1]
                            pred_test = res['model'].predict_proba(X_test)[:, 1]
                            
                        meta_X_val[name] = pred_val
                        meta_X_test[name] = pred_test
                    
                    # Treinar Meta-Learner
                    meta_model = LogisticRegression(random_state=42)
                    meta_model.fit(meta_X_val, y_val)
                    
                    # Avaliar
                    y_prob_val = meta_model.predict_proba(meta_X_val)[:, 1]
                    best_thr = 0.5 # Simplificação, poderia otimizar também
                    y_pred_val = (y_prob_val >= best_thr).astype(int)
                    metrics_val = calculate_classification_metrics(y_val, y_pred_val)
                    
                    y_prob_test = meta_model.predict_proba(meta_X_test)[:, 1]
                    y_pred_test = (y_prob_test >= best_thr).astype(int)
                    metrics_test = calculate_classification_metrics(y_test, y_pred_test)
                    
                    # Criar Wrapper
                    from src.models.ensemble import StackingEnsembleWrapper
                    
                    # Preparar base_models dict
                    base_models_dict = {}
                    for name, res in model_results.items():
                        base_models_dict[name] = res['calibrador'] if res['calibrador'] else res['model']
                        
                    ensemble_wrapper = StackingEnsembleWrapper(meta_model, base_models_dict)
                    
                    model_results['StackingEnsemble'] = {
                        'model': ensemble_wrapper,
                        'calibrador': None, # Wrapper já retorna proba
                        'best_threshold': best_thr,
                        'metrics_val': metrics_val,
                        'metrics_test': metrics_test
                    }
                    logger.info(f"Stacking Ensemble finalizado. F1-Val: {metrics_val['F1-Score']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Falha no Stacking Ensemble: {e}")
            
            # Selecionar campeão por F1 em validação
            best_name = max(
                model_results,
                key=lambda n: model_results[n]['metrics_val'].get('F1-Score', -1)
            )
            
            champion = model_results[best_name]
            champion['model_name'] = best_name
            champion['horizon_days'] = h
            champion['all_results'] = model_results
            
            results[h] = champion
            
            logger.info(
                f"🏆 CAMPEÃO: {best_name} | "
                f"F1-VAL={champion['metrics_val']['F1-Score']:.3f} | "
                f"AUC-VAL={champion['metrics_val'].get('AUC-ROC', 0):.3f} | "
                f"F1-TEST={champion['metrics_test']['F1-Score']:.3f}"
            )
        
        return results
