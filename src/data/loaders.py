"""
src/data/loaders.py
Carregamento de dados Excel/CSV com mapeamento preservado do sistema legado.

IMPORTANTE: Mantém compatibilidade com estrutura de dados existente.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import unicodedata
import re

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def standardize_string(text: str) -> str:
    """
    Padroniza string: remove acentos, converte para minúsculas, remove espaços extras.
    
    Args:
        text: String a padronizar
        
    Returns:
        String padronizada
        
    Example:
        >>> standardize_string("Data e Hora de Início")
        'data_e_hora_de_inicio'
    """
    if not isinstance(text, str):
        return str(text).lower().strip()
    
    # Remover acentos
    nfd_form = unicodedata.normalize('NFD', text)
    text_without_accents = ''.join(char for char in nfd_form if unicodedata.category(char) != 'Mn')
    
    # Minúsculas e substituir espaços por underscores
    text_clean = text_without_accents.lower().strip()
    text_clean = re.sub(r'[\s/\\]+', '_', text_clean)
    text_clean = re.sub(r'[^\w_]', '', text_clean)
    
    return text_clean


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza nomes de colunas do DataFrame.
    
    Args:
        df: DataFrame com colunas originais
        
    Returns:
        DataFrame com colunas padronizadas
    """
    df = df.copy()
    df.columns = [standardize_string(col) for col in df.columns]
    return df


def load_falhas_excel(
    uploaded_file,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Carrega arquivo Excel de ocorrências de falhas.
    
    MAPEAMENTO PRESERVADO DO SISTEMA LEGADO:
    - Mapeamento por nome exato de colunas
    - Criação de ativo_unico = {instalacao} - {modulo_envolvido} - {ativo}
    
    Args:
        uploaded_file: Arquivo uploaded (Streamlit) ou caminho (Path)
        config: Configuração do sistema (opcional)
        
    Returns:
        DataFrame padronizado com colunas:
        - ativo_unico: str (identificador único)
        - data: datetime (timestamp da falha)
        - instalacao: str
        - modulo_envolvido: str
        - ativo: str
        - regional: str
        - tipo_manutencao: str
        - descricao: str
        - id_evento: str
        
    Raises:
        ValueError: Se colunas essenciais não encontradas
    """
    logger.info("Carregando arquivo de falhas...")
    
    # Carregar Excel
    try:
        if hasattr(uploaded_file, 'read'):
            # Streamlit UploadedFile
            df = pd.read_excel(uploaded_file)
        else:
            # Path
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        logger.error(f"Erro ao ler Excel: {e}")
        raise
    
    logger.info(f"Arquivo carregado: {len(df)} linhas, {len(df.columns)} colunas")
    
    # Padronizar nomes de colunas
    df = standardize_columns(df)
    
    # Mapeamento de colunas (preservado do sistema legado)
    column_mapping = {
        'data_e_hora_de_inicio': 'data',
        'data_hora_inicio': 'data',
        'data_inicio': 'data',
        'equipamento_componente_envolvido': 'ativo',
        'equipamento': 'ativo',
        'componente': 'ativo',
        'instalacao_localizacao': 'instalacao',
        'instalacao': 'instalacao',
        'localizacao': 'instalacao',
        'modulo_envolvido': 'modulo_envolvido',
        'modulo': 'modulo_envolvido',
        'regional': 'regional',
        'tipo_de_ocorrencia': 'tipo_manutencao',
        'tipo_ocorrencia': 'tipo_manutencao',
        'tipo': 'tipo_manutencao',
        'descricao_do_evento': 'descricao',
        'descricao': 'descricao',
        'id_evento': 'id_evento',
        'id': 'id_evento'
    }
    
    # Aplicar mapeamento
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Verificar colunas essenciais
    required_cols = ['data', 'ativo', 'instalacao', 'modulo_envolvido']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Colunas essenciais faltando: {missing_cols}")
        logger.info(f"Colunas disponíveis: {list(df.columns)}")
        raise ValueError(
            f"Colunas essenciais não encontradas: {missing_cols}.\n"
            f"Verifique se o arquivo possui as colunas corretas."
        )
    
    # Converter data
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    
    # Remover linhas com data inválida
    invalid_dates = df['data'].isnull().sum()
    if invalid_dates > 0:
        logger.warning(f"Removendo {invalid_dates} linhas com data inválida")
        df = df[df['data'].notna()].copy()
    
    # Limpar strings
    string_cols = ['ativo', 'instalacao', 'modulo_envolvido']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Criar ativo_unico (CRÍTICO - preservado do legado)
    # Formato: {instalacao} - {modulo_envolvido} - {ativo}
    df['ativo_unico'] = (
        df['instalacao'].astype(str) + ' - ' +
        df['modulo_envolvido'].astype(str) + ' - ' +
        df['ativo'].astype(str)
    )
    
    # Preencher colunas opcionais se não existirem
    if 'regional' not in df.columns:
        df['regional'] = 'N/A'
    
    if 'tipo_manutencao' not in df.columns:
        df['tipo_manutencao'] = 'Corretiva'
    
    if 'descricao' not in df.columns:
        df['descricao'] = ''
    
    if 'id_evento' not in df.columns:
        # Criar ID baseado em índice
        df['id_evento'] = [f"EV{i:06d}" for i in range(len(df))]
    
    # Selecionar colunas finais
    final_cols = [
        'ativo_unico', 'data', 'instalacao', 'modulo_envolvido', 'ativo',
        'regional', 'tipo_manutencao', 'descricao', 'id_evento'
    ]
    
    df_final = df[final_cols].copy()
    
    # Ordenar por ativo e data
    df_final = df_final.sort_values(['ativo_unico', 'data']).reset_index(drop=True)
    
    logger.info(f"✓ Dados de falhas carregados: {len(df_final)} registros, {df_final['ativo_unico'].nunique()} ativos únicos")
    
    return df_final


def load_pcm_excel(
    uploaded_file,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Carrega arquivo Excel de Plano de Controle de Manutenção (PCM).
    
    MAPEAMENTO PRESERVADO DO SISTEMA LEGADO:
    - Mapeamento por ÍNDICE de colunas (posições fixas 18, 19, 20, 29)
    - Fallback para mapeamento por nome
    
    Args:
        uploaded_file: Arquivo uploaded ou caminho
        config: Configuração do sistema (opcional)
        
    Returns:
        DataFrame padronizado com colunas:
        - ativo_unico: str
        - data_de_abertura: datetime
        - tipo_manutencao: str
        - descricao: str
    """
    logger.info("Carregando arquivo PCM...")
    
    # Carregar Excel
    try:
        if hasattr(uploaded_file, 'read'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        logger.error(f"Erro ao ler Excel PCM: {e}")
        raise
    
    logger.info(f"Arquivo PCM carregado: {len(df)} linhas, {len(df.columns)} colunas")
    
    # MAPEAMENTO POR ÍNDICE (preservado do legado)
    # Tentativa 1: Usar índices fixos conhecidos
    try:
        if len(df.columns) > 29:
            df_mapped = pd.DataFrame({
                'ativo': df.iloc[:, 18],  # Índice 18
                'instalacao': df.iloc[:, 19],  # Índice 19
                'modulo_envolvido': df.iloc[:, 20],  # Índice 20
                'data_de_abertura': df.iloc[:, 29]  # Índice 29
            })
            logger.info("Mapeamento por índice aplicado com sucesso")
        else:
            raise ValueError("Menos colunas que esperado, tentando mapeamento por nome")
    except Exception as e:
        logger.warning(f"Mapeamento por índice falhou: {e}. Tentando por nome...")
        
        # Fallback: Mapeamento por nome
        df = standardize_columns(df)
        
        column_mapping = {
            'equipamento': 'ativo',
            'instalacao': 'instalacao',
            'modulo': 'modulo_envolvido',
            'data_abertura': 'data_de_abertura',
            'data': 'data_de_abertura'
        }
        
        df_mapped = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df_mapped.columns and new_col not in df_mapped.columns:
                df_mapped[new_col] = df_mapped[old_col]
    
    # Verificar colunas essenciais
    required_cols = ['ativo', 'instalacao', 'modulo_envolvido', 'data_de_abertura']
    missing_cols = [col for col in required_cols if col not in df_mapped.columns]
    
    if missing_cols:
        logger.error(f"Colunas PCM faltando: {missing_cols}")
        raise ValueError(f"Colunas PCM não encontradas: {missing_cols}")
    
    # Converter data
    df_mapped['data_de_abertura'] = pd.to_datetime(df_mapped['data_de_abertura'], errors='coerce')
    
    # Criar ativo_unico
    df_mapped['ativo_unico'] = (
        df_mapped['instalacao'].astype(str) + ' - ' +
        df_mapped['modulo_envolvido'].astype(str) + ' - ' +
        df_mapped['ativo'].astype(str)
    )
    
    # Adicionar colunas opcionais
    if 'tipo_manutencao' not in df_mapped.columns:
        df_mapped['tipo_manutencao'] = 'Preventiva'
    
    if 'descricao' not in df_mapped.columns:
        df_mapped['descricao'] = 'Manutenção preventiva programada'
    
    # Selecionar colunas finais
    final_cols = ['ativo_unico', 'data_de_abertura', 'tipo_manutencao', 'descricao']
    df_final = df_mapped[final_cols].copy()
    
    # Remover linhas com data inválida
    df_final = df_final[df_final['data_de_abertura'].notna()].copy()
    
    logger.info(f"✓ Dados PCM carregados: {len(df_final)} registros")
    
    return df_final


def load_csv(
    file_path: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """
    Carrega arquivo CSV com encoding automático.
    
    Args:
        file_path: Caminho do arquivo CSV
        **kwargs: Argumentos adicionais para pd.read_csv
        
    Returns:
        DataFrame carregado
    """
    logger.info(f"Carregando CSV: {file_path}")
    
    # Tentar diferentes encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            logger.info(f"CSV carregado com encoding {encoding}: {len(df)} linhas")
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Não foi possível ler o arquivo {file_path} com encodings: {encodings}")
