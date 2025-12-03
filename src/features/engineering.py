"""
src/features/engineering.py
Pipeline de Engenharia de Features para Manutenção Preditiva.

IMPORTANTE: Gera features temporais, estatísticas, sazonais e interações
para alimentar modelos de Machine Learning.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FeatureEngineeringPipeline:
    """
    Pipeline completo de engenharia de features para análise preditiva.
    
    Features geradas:
    - Temporais: TBF, idade do ativo, falhas acumuladas
    - Estatísticas: médias e desvios móveis de TBF
    - Sazonais: mês, dia da semana, componentes cíclicas (sin/cos)
    - Interações: ratios, distâncias de min/max, volatilidade
    
    Args:
        lags: Lista de lags para features temporais (ex: [1, 3, 6, 12])
        rolling_windows: Lista de janelas para médias móveis (ex: [3, 6, 12])
        include_sazonalidade: Se True, inclui features cíclicas de sazonalidade
        include_interacoes: Se True, inclui features de interação
    
    Example:
        >>> pipeline = FeatureEngineeringPipeline(
        ...     lags=[1, 3, 6],
        ...     rolling_windows=[3, 6],
        ...     include_sazonalidade=True
        ... )
        >>> df_features = pipeline.fit_transform(df_raw)
        >>> print(f"Features criadas: {len(pipeline.get_feature_names())}")
    """
    
    def __init__(
        self,
        lags: List[int] = [1, 3, 6, 12],
        rolling_windows: List[int] = [3, 6, 12],
        include_sazonalidade: bool = True,
        include_interacoes: bool = True
    ):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.include_sazonalidade = include_sazonalidade
        self.include_interacoes = include_interacoes
        self.feature_names_ = []
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todo o pipeline de features.
        
        Args:
            df: DataFrame com colunas 'ativo_unico' e 'data'
            
        Returns:
            DataFrame com features adicionadas
            
        Raises:
            ValueError: Se colunas obrigatórias estiverem ausentes
        """
        # Validar colunas essenciais
        required = ['ativo_unico', 'data']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Colunas obrigatórias: {required}")
        
        logger.info(f"Iniciando Feature Engineering: {len(df)} registros")
        
        # Ordenação temporal (CRÍTICO para lags e rolling)
        df = df.sort_values(by=['ativo_unico', 'data']).copy()
        
        # Etapas sequenciais
        df = self._criar_features_temporais(df)
        df = self._criar_features_estatisticas(df)
        df = self._criar_features_tendencias(df)
        
        if self.include_sazonalidade:
            df = self._criar_sazonalidade_ciclica(df)
        
        if self.include_interacoes:
            df = self._criar_interacoes(df)
            
        # Features de Confiabilidade (Sinergia)
        df = self._criar_features_confiabilidade(df)
        
        # Tratamento de valores faltantes
        df = self._tratar_valores_faltantes(df)
        
        # Extrair nomes de features (excluir metadados)
        metadata_cols = ['ativo_unico', 'data', 'instalacao', 'modulo_envolvido', 
                        'ativo', 'regional', 'descricao', 'id_evento', 'tipo_manutencao']
        self.feature_names_ = [c for c in df.columns if c not in metadata_cols 
                              and not c.startswith('falha_em_')]
        
        logger.info(f"Feature Engineering concluído: {len(self.feature_names_)} features criadas")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de nomes de features geradas."""
        return self.feature_names_
    
    def _criar_features_temporais(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features temporais base: TBF, falhas acumuladas, idade do ativo.
        
        TBF (Time Between Failures): Dias desde última falha do mesmo ativo.
        """
        logger.info("Criando features temporais...")
        
        # TBF: Tempo desde última falha (por ativo)
        df['tbf'] = df.groupby('ativo_unico')['data'].diff().dt.days
        
        # Falhas acumuladas até o momento (por ativo)
        df['falhas_acumuladas'] = df.groupby('ativo_unico').cumcount() + 1
        
        # Idade do ativo (dias desde primeira falha registrada)
        df['primeira_falha'] = df.groupby('ativo_unico')['data'].transform('first')
        df['idade_ativo_dias'] = (df['data'] - df['primeira_falha']).dt.days
        df = df.drop(columns=['primeira_falha'])
        
        # Features temporais extraídas da data
        df['mes_falha'] = df['data'].dt.month
        df['dia_semana_falha'] = df['data'].dt.dayofweek
        df['ano_falha'] = df['data'].dt.year
        df['trimestre_falha'] = df['data'].dt.quarter
        
        logger.info("✓ Features temporais criadas: TBF, idade, data extracts")
        
        return df
    
    def _criar_features_estatisticas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria estatísticas móveis: médias, desvios, min, max de TBF.
        """
        logger.info("Criando features estatísticas...")
        
        for window in self.rolling_windows:
            # Média móvel
            df[f'tbf_mean_{window}eventos'] = (
                df.groupby('ativo_unico')['tbf']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            
            # Desvio padrão móvel
            df[f'tbf_std_{window}eventos'] = (
                df.groupby('ativo_unico')['tbf']
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
            
            # Mínimo móvel
            df[f'tbf_min_{window}eventos'] = (
                df.groupby('ativo_unico')['tbf']
                .transform(lambda x: x.rolling(window, min_periods=1).min())
            )
            
            # Máximo móvel
            df[f'tbf_max_{window}eventos'] = (
                df.groupby('ativo_unico')['tbf']
                .transform(lambda x: x.rolling(window, min_periods=1).max())
            )
        
        logger.info(f"✓ Features estatísticas criadas: {4 * len(self.rolling_windows)} colunas")
        
        return df
    
    def _criar_features_tendencias(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de tendência: variação percentual, volatilidade.
        """
        logger.info("Criando features de tendências...")
        
        # Variação percentual de TBF (1 lag)
        df['tbf_pct_change_1m'] = (
            df.groupby('ativo_unico')['tbf']
            .transform(lambda x: x.pct_change(periods=1))
        )
        
        # Volatilidade de TBF (3 eventos = ~3 meses se aproximadamente mensal)
        df['tbf_volatilidade_3m'] = (
            df.groupby('ativo_unico')['tbf']
            .transform(lambda x: x.rolling(3, min_periods=1).std() / (x.rolling(3, min_periods=1).mean() + 1e-6))
        )
        
        logger.info("✓ Features de tendências criadas")
        
        return df
    
    def _criar_sazonalidade_ciclica(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features cíclicas para capturar sazonalidade.
        
        Usa sin/cos para preservar continuidade circular (dezembro → janeiro).
        """
        logger.info("Criando features de sazonalidade cíclica...")
        
        # Mês cíclico (1-12)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes_falha'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes_falha'] / 12)
        
        # Trimestre cíclico (1-4)
        df['trimestre_sin'] = np.sin(2 * np.pi * df['trimestre_falha'] / 4)
        df['trimestre_cos'] = np.cos(2 * np.pi * df['trimestre_falha'] / 4)
        
        logger.info("✓ Features de sazonalidade cíclica criadas")
        
        return df
    
    def _criar_interacoes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de interação: ratios, distâncias.
        """
        logger.info("Criando features de interação...")
        
        # Ratio TBF atual vs média (3, 6 eventos)
        if 'tbf_mean_3eventos' in df.columns:
            df['ratio_vs_mean_3m'] = df['tbf'] / (df['tbf_mean_3eventos'] + 1e-6)
        
        if 'tbf_mean_6eventos' in df.columns:
            df['ratio_vs_mean_6m'] = df['tbf'] / (df['tbf_mean_6eventos'] + 1e-6)
        
        # Distância do mínimo recente
        if 'tbf_min_6eventos' in df.columns:
            df['dist_from_min_6m'] = df['tbf'] - df['tbf_min_6eventos']
        
        # Distância do máximo recente
        if 'tbf_max_6eventos' in df.columns:
            df['dist_from_max_6m'] = df['tbf_max_6eventos'] - df['tbf']
        
        # Normalização TBF (z-score sobre rolling)
        if 'tbf_mean_6eventos' in df.columns and 'tbf_std_6eventos' in df.columns:
            df['tbf_zscore_6m'] = (df['tbf'] - df['tbf_mean_6eventos']) / (df['tbf_std_6eventos'] + 1e-6)
        
        logger.info("✓ Features de interação criadas")
        
        return df
    
    def _criar_features_confiabilidade(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features baseadas em análise de confiabilidade (Sobrevivência).
        Gera: prob_sobrevivencia_30d (Probabilidade de não falhar nos próximos 30 dias dado TBF atual)
        """
        logger.info("Criando features de confiabilidade...")
        try:
            from src.models.reliability import ReliabilityAnalyzer
            
            # Ajustar modelo global (simplificação para v2.0)
            # Idealmente seria por família de ativos
            analyzer = ReliabilityAnalyzer()
            analyzer.fit(df, time_col='tbf')
            
            if analyzer.wf:
                # Vetorizar predição usando WeibullFitter diretamente se possível
                # P(T > t + 30 | T > t) = S(t + 30) / S(t)
                
                t_current = df['tbf'].fillna(0).values
                
                # Calcular S(t) e S(t+30)
                # Weibull survival: exp(-(t/lambda)^rho)
                lambda_ = analyzer.wf.lambda_
                rho_ = analyzer.wf.rho_
                
                s_t = np.exp(-(t_current / lambda_) ** rho_)
                s_t_30 = np.exp(-((t_current + 30) / lambda_) ** rho_)
                
                df['prob_sobrevivencia_30d'] = s_t_30 / (s_t + 1e-6)
                logger.info("✓ Feature prob_sobrevivencia_30d criada (Weibull)")
                
            else:
                df['prob_sobrevivencia_30d'] = 0.5
                logger.warning("Modelo de confiabilidade não ajustado. Usando default 0.5")
                
        except Exception as e:
            logger.error(f"Erro ao criar features de confiabilidade: {e}")
            df['prob_sobrevivencia_30d'] = 0.5
            
        return df
    
    def _tratar_valores_faltantes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata valores faltantes de forma conservadora.
        
        Estratégia:
        - NaN em TBF (primeira ocorrência) → preencher com mediana do ativo
        - NaN em features estatísticas → preencher com 0
        """
        logger.info("Tratando valores faltantes...")
        
        # TBF: preencher com mediana do ativo
        if 'tbf' in df.columns:
            df['tbf'] = df.groupby('ativo_unico')['tbf'].transform(
                lambda x: x.fillna(x.median() if x.median() == x.median() else 0)  # Check for NaN median
            )
        
        # Demais features numéricas: preencher com 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Verificar se ainda há nulos
        nulls_remaining = df.isnull().sum().sum()
        if nulls_remaining > 0:
            logger.warning(f"Ainda restam {nulls_remaining} valores nulos após tratamento")
        else:
            logger.info("✓ Todos os valores faltantes tratados")
        
        return df
