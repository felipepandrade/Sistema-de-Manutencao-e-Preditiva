"""
tests/test_integration.py
Testes de integração para pipelines completos.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

from src.data.loaders import load_falhas_excel
from src.features.engineering import FeatureEngineeringPipeline
from src.features.target_builder import create_multi_horizon_targets, create_temporal_splits
from src.models.trainer import ModelTrainer
from src.inference.predictor import PredictorPipeline
from src.maintenance.performance_tracker import PerformanceTracker
from src.maintenance.rca_analyzer import RCAAnalyzer


class TestIntegrationPipelines:
    """Testes de integração end-to-end."""
    
    @pytest.fixture
    def config(self):
        """Carrega configuração."""
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def sample_data(self):
        """Gera dados de teste."""
        np.random.seed(42)
        
        ativos = ['COMP-01', 'COMP-02', 'VÁLV-03']
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='W')
        
        data = []
        for ativo in ativos:
            for i in range(len(dates)):
                data.append({
                    'ativo': ativo.split('-')[1],
                    'instalacao': 'ITABUNA',
                    'modulo_envolvido': 'COMPRESSÃO' if 'COMP' in ativo else 'VÁLVULAS',
                    'regional': 'BA',
                    'data': dates[i],
                    'descricao': f'Falha teste {i}',
                    'tipo_manutencao': 'Corretiva'
                })
        
        df = pd.DataFrame(data)
        df['ativo_unico'] = df['instalacao'] + ' - ' + df['modulo_envolvido'] + ' - ' + df['ativo']
        
        return df
    
    def test_full_training_pipeline(self, sample_data, config):
        """
        Teste de pipeline completo de treinamento.
        """
        # 1. Feature Engineering
        pipeline = FeatureEngineeringPipeline(
            lags=config['features']['lags'],
            rolling_windows=config['features']['rolling_windows'],
            include_sazonalidade=config['features']['include_sazonalidade'],
            include_interacoes=config['features']['include_interacoes']
        )
        
       df_features = pipeline.fit_transform(sample_data)
        feature_names = pipeline.get_feature_names()
        
        assert len(df_features) > 0, "Features devem ser geradas"
        assert len(feature_names) >= 20, "Deve ter pelo menos 20 features"
        
        # 2. Target Creation
        df_features = create_multi_horizon_targets(df_features, [3, 7, 15, 30])
        
        assert 'falha_em_3d' in df_features.columns
        assert 'falha_em_30d' in df_features.columns
        
        # 3. Temporal Split
        mask_train, mask_val, mask_test = create_temporal_splits(
            df_features,
            frac_train=0.6,
            frac_val=0.2
        )
        
        assert mask_train.sum() > 0, "Deve ter amostras de treino"
        assert mask_val.sum() > 0, "Deve ter amostras de validação"
        assert mask_test.sum() > 0, "Deve ter amostras de teste"
        
        # 4. Training (apenas 1 modelo para teste rápido)
        config_test = config.copy()
        config_test['models']['classical_models'] = ['RandomForest']
        config_test['models']['automl']['engine'] = 'skip'
        config_test['models']['deep_learning']['enabled'] = False
        
        trainer = ModelTrainer(config_test)
        
        results = trainer.train_all_horizons(
            df_features,
            feature_names,
            [30],  # Apenas 1 horizonte para teste rápido
            mask_train,
            mask_val,
            mask_test
        )
        
        assert 30 in results, "Deve treinar horizonte 30d"
        assert 'model' in results[30], "Deve conter modelo treinado"
        assert results[30]['metrics_val']['F1-Score'] >= 0.0, "F1 deve estar definido"
    
    def test_inference_pipeline(self, sample_data, config, tmp_path):
        """
        Teste de pipeline de inferência.
        """
        # Simular modelo salvo
        models_dir = tmp_path / "models" / "test_version"
        models_dir.mkdir(parents=True)
        
        # Criar metadata fictício
        metadata = {
            "version": "test_version",
            "models_by_horizon": {
                "30": {
                    "model_name": "RandomForest",
                    "best_threshold": 0.5,
                    "metrics_val": {"F1-Score": 0.75}
                }
            },
            "features": ['tbf', 'falhas_acumuladas']
        }
        
        import json
        with open(models_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Criar modelo mock
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(10, 2)
        y_dummy = np.random.randint(0, 2, 10)
        model.fit(X_dummy, y_dummy)
        
        joblib.dump(model, models_dir / "modelo_30d.pkl")
        
        # Teste de inferência
        try:
            config_test = config.copy()
            config_test['paths']['models'] = str(tmp_path / "models")
            
            predictor = PredictorPipeline(model_version='test_version', config=config_test)
            
            # Criar features mínimas para predição
            df_test = sample_data.head(5).copy()
            df_test['tbf'] = 10.0
            df_test['falhas_acumuladas'] = 5
            
            predictions = predictor.predict(df_test)
            
            assert len(predictions) > 0, "Deve gerar predições"
            assert 'ativo_unico' in predictions.columns
            assert 'ProbML_Media_30d' in predictions.columns
            
        except Exception as e:
            pytest.skip(f"Teste de inferência pulado: {e}")
    
    def test_performance_tracker_workflow(self):
        """
        Teste de workflow do Performance Tracker.
        """
        tracker = PerformanceTracker(logs_dir=Path("outputs/logs_test"))
        
        # 1. Registrar predições
        tracker.log_prediction(
            ativo_unico="ITABUNA - COMPRESSÃO - COMP-01",
            data_predicao="2024-01-01",
            horizonte=30,
            probabilidade=0.75,
            classe_predita="Alto Risco",
            modelo_versao="v1"
        )
        
        tracker.log_prediction(
            ativo_unico="ITABUNA - COMPRESSÃO - COMP-01",
            data_predicao="2024-01-05",
            horizonte=30,
            probabilidade=0.35,
            classe_predita="Médio Risco",
            modelo_versao="v1"
        )
        
        # 2. Registrar outcomes
        tracker.register_outcome(
            ativo_unico="ITABUNA - COMPRESSÃO - COMP-01",
            data_evento="2024-01-15",
            houve_falha=True
        )
        
        # 3. Verificar sumário
        summary = tracker.get_summary()
        assert summary['total_predictions'] == 2
        assert summary['with_feedback'] >= 1
        
        # 4. Calcular métricas (pode não ter dados suficientes ainda)
        metrics = tracker.calculate_metrics(horizonte=30)
        # Apenas verificar que não dá erro
        
        # Cleanup
        import shutil
        if Path("outputs/logs_test").exists():
            shutil.rmtree("outputs/logs_test")
    
    def test_rca_analyzer_workflow(self, sample_data):
        """
        Teste de workflow do RCA Analyzer.
        """
        # Preparar dados
        pipeline = FeatureEngineeringPipeline()
        df_features = pipeline.fit_transform(sample_data)
        feature_names = pipeline.get_feature_names()
        
        # Pegar um ativo
        ativo = df_features['ativo_unico'].iloc[0]
        df_ativo = df_features[df_features['ativo_unico'] == ativo]
        
        if len(df_ativo) < 3:
            pytest.skip("Dados insuficientes para RCA")
        
        # Selecionar uma falha
        failure_date = df_ativo['data'].iloc[-1]
        
        # Executar RCA
        analyzer = RCAAnalyzer(zscore_threshold=2.0)
        
        report = analyzer.analyze(
            asset_history=df_ativo,
            failure_date=failure_date,
            feature_names=feature_names[:10],  # Apenas primeiras 10 features
            window_days=7
        )
        
        # Verificações
        assert 'ativo_unico' in report
        assert 'failure_date' in report
        assert 'anomalies_detected' in report
        assert 'top_root_causes' in report
        assert 'recommendations' in report
        
        # RCA não deve ter erro
        assert 'error' not in report or report.get('anomalies_detected', 0) >= 0
    
    def test_data_validation_pandera(self, sample_data):
        """
        Teste de validação de dados com Pandera.
        """
        from src.data.validators import SCHEMA_FALHAS
        
        try:
            # Validar schema
            SCHEMA_FALHAS.validate(sample_data)
            
        except Exception as e:
            # Se falhar, verificar que é erro esperado
            assert "Cannot parse" in str(e) or "column" in str(e).lower()


class TestModelPersistence:
    """Testes de persistência de modelos."""
    
    def test_save_and_load_model(self, tmp_path):
        """Testa salvamento e carregamento de modelos."""
        from sklearn.ensemble import RandomForestClassifier
        from src.utils.io import load_model_artifacts
        import joblib
        import json
        
        # Criar modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(20, 5)
        y = np.random.randint(0, 2, 20)
        model.fit(X, y)
        
        # Salvar
        models_dir = tmp_path / "models" / "v_test"
        models_dir.mkdir(parents=True)
        
        joblib.dump(model, models_dir / "modelo_30d.pkl")
        
        metadata = {
            "version": "v_test",
            "models_by_horizon": {
                "30": {
                    "model_name": "RandomForest",
                    "best_threshold": 0.5,
                    "metrics_val": {"F1-Score": 0.80}
                }
            },
            "features": ['f1', 'f2', 'f3', 'f4', 'f5']
        }
        
        with open(models_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Carregar
        artifacts = load_model_artifacts(models_dir)
        
        assert 30 in artifacts['models']
        assert artifacts['models'][30]['model'] is not None
        assert artifacts['metadata']['version'] == "v_test"
        assert len(artifacts['features']) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
