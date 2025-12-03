"""
tests/test_core.py
Testes unitários e de integração para o core do sistema.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os

from src.data.loaders import load_falhas_excel
from src.features.engineering import FeatureEngineeringPipeline
from src.features.target_builder import create_multi_horizon_targets
from src.models.classical import montar_modelos_base

# Fixtures
@pytest.fixture
def sample_data():
    """Cria dados fake para teste."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'data': dates,
        'ativo': ['COMP-01'] * 100,
        'instalacao': ['TESTE'] * 100,
        'modulo_envolvido': ['MOD-A'] * 100,
        'regional': ['BA'] * 100,
        'tipo_manutencao': ['Corretiva'] * 100,
        'descricao': ['Falha teste'] * 100,
        'id_evento': range(100)
    })
    # Criar ativo_unico
    df['ativo_unico'] = df['instalacao'] + ' - ' + df['modulo_envolvido'] + ' - ' + df['ativo']
    return df

@pytest.fixture
def config():
    return {
        'features': {
            'lags': [1],
            'rolling_windows': [3],
            'include_sazonalidade': True,
            'include_interacoes': False
        },
        'models': {
            'horizontes': [3],
            'calibration': {'method': 'sigmoid'}
        }
    }

# Testes Data Layer
def test_data_loader_structure(sample_data):
    """Testa se colunas essenciais existem."""
    required = ['ativo_unico', 'data']
    for col in required:
        assert col in sample_data.columns

# Testes Feature Engineering
def test_feature_engineering(sample_data, config):
    """Testa pipeline de features."""
    pipeline = FeatureEngineeringPipeline(**config['features'])
    df_feat = pipeline.fit_transform(sample_data)
    
    # Verificar se features foram criadas
    assert 'tbf' in df_feat.columns
    assert 'tbf_mean_3eventos' in df_feat.columns
    assert 'mes_sin' in df_feat.columns
    assert len(df_feat) == len(sample_data)

# Testes Target Builder
def test_target_creation(sample_data, config):
    """Testa criação de targets."""
    pipeline = FeatureEngineeringPipeline(**config['features'])
    df_feat = pipeline.fit_transform(sample_data)
    
    df_targets = create_multi_horizon_targets(df_feat, config['models']['horizontes'])
    
    assert 'falha_em_3d' in df_targets.columns
    # Verificar se é binário
    assert df_targets['falha_em_3d'].isin([0, 1]).all()

# Testes Models
def test_model_factory():
    """Testa criação de modelos."""
    models = montar_modelos_base()
    assert 'RandomForest' in models
    assert 'XGBoost' in models

# Testes Integração
def test_full_pipeline_run(sample_data, config, tmp_path):
    """Testa execução ponta a ponta simplificada."""
    # 1. Features
    pipeline = FeatureEngineeringPipeline(**config['features'])
    df_feat = pipeline.fit_transform(sample_data)
    
    # 2. Targets
    df_targets = create_multi_horizon_targets(df_feat, [3])
    
    # 3. Treino (mock)
    X = df_targets[['tbf', 'tbf_mean_3eventos']].fillna(0)
    y = df_targets['falha_em_3d']
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    
    # 4. Predição
    preds = model.predict(X)
    assert len(preds) == len(X)
