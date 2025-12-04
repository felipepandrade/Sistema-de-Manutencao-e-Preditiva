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
        'equipamentocomponente_envolvido': 'ativo',  # SEM underscore (variação do arquivo original)
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
        'id': 'id_evento',
        'prioridade': 'prioridade'  # Campo adicional do script original
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
    
    # Adicionar prioridade se existir
    if 'prioridade' in df.columns:
        final_cols.append('prioridade')
    
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
    - Padronização de colunas ANTES do mapeamento
    - Mapeamento por ÍNDICE de colunas (posições fixas 18, 19, 20, 29)
    - Mapeamento geral adicional
    - Fallback robusto para criação de ativo_unico
    
    Args:uploaded_file: Arquivo uploaded ou caminho
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
    
    # Guardar colunas originais para mapeamento por índice
    original_cols = df.columns.tolist()
    
    # MAPEAMENTO POR ÍNDICE com PADRONIZAÇÃO (do script original)
    try:
        if len(original_cols) > 29:
            # Criar mapeamento dinâmico padronizando os nomes
            rename_map_dinamico = {
                standardize_string(original_cols[19]): 'instalacao',  # Índice 19
                standardize_string(original_cols[20]): 'sistema',     # Índice 20
                standardize_string(original_cols[18]): 'local_de_servico',  # Índice 18
                standardize_string(original_cols[29]): 'descricao_do_equipamento'  # Índice 29
            }
            
            # Padronizar TODAS as colunas
            df = standardize_columns(df)
            
            # Aplicar mapeamento dinâmico
            df.rename(columns=rename_map_dinamico, inplace=True)
            
            logger.info("Mapeamento PCM por índice (com padronização) aplicado")
            df_mapped = df.copy()
        else:
            raise ValueError("Menos de 30 colunas, tentando mapeamento por nome")
            
    except Exception as e:
        logger.warning(f"Mapeamento PCM por índice falhou: {e}. Tentando por nome...")
        
        # Fallback: Mapeamento por nome
        df = standardize_columns(df)
        df_mapped = df.copy()
    
    # Mapeamento geral adicional (do script original)
    rename_map_geral = {
        'tipo': 'tipo_manutencao',
        'data_da_solicitacao': 'data_de_abertura',
        'data_abertura': 'data_de_abertura',
        'data_solicitacao': 'data_de_abertura',
        'equipamento': 'equipamento_original_j',
        'status': 'status_da_ordem',
        'descricao': 'descricao'
    }
    
    # Aplicar apenas colunas existentes
    actual_rename = {k: v for k, v in rename_map_geral.items() if k in df_mapped.columns}
    df_mapped.rename(columns=actual_rename, inplace=True)
    
    # Criar ativo_unico com FALLBACKS robustos (do script original)
    if all(col in df_mapped.columns for col in ['instalacao', 'sistema', 'descricao_do_equipamento']):
        # Método 1: Instalação + Sistema + Descrição
        df_mapped['instalacao'] = df_mapped['instalacao'].astype(str).fillna('')
        df_mapped['sistema'] = df_mapped['sistema'].astype(str).fillna('')
        df_mapped['descricao_do_equipamento'] = df_mapped['descricao_do_equipamento'].astype(str).fillna('')
        df_mapped['ativo_unico'] = (
            df_mapped['instalacao'] + ' - ' +
            df_mapped['sistema'] + ' - ' +
            df_mapped['descricao_do_equipamento']
        )
        # Aliases
        df_mapped['modulo_envolvido'] = df_mapped['sistema']
        df_mapped['ativo'] = df_mapped['descricao_do_equipamento']
        
    elif all(col in df_mapped.columns for col in ['sistema', 'descricao_do_equipamento']):
        # Fallback 1: Sem instalação
        df_mapped['sistema'] = df_mapped['sistema'].astype(str).fillna('')
        df_mapped['descricao_do_equipamento'] = df_mapped['descricao_do_equipamento'].astype(str).fillna('')
        df_mapped['ativo_unico'] = (
            df_mapped['sistema'] + ' - ' +
            df_mapped['descricao_do_equipamento']
        )
        df_mapped['instalacao'] = 'N/A'
        df_mapped['modulo_envolvido'] = df_mapped['sistema']
        df_mapped['ativo'] = df_mapped['descricao_do_equipamento']
        
    elif all(col in df_mapped.columns for col in ['instalacao', 'modulo_envolvido', 'ativo']):
        # Fallback 2: Colunas já padronizadas
        df_mapped['ativo_unico'] = (
            df_mapped['instalacao'].astype(str) + ' - ' +
            df_mapped['modulo_envolvido'].astype(str) + ' - ' +
            df_mapped['ativo'].astype(str)
        )
    else:
        logger.error(f"Estrutura PCM não reconhecida. Colunas: {list(df_mapped.columns)}")
        raise ValueError("Não foi possível criar ativo_unico para PCM")
    
    # Converter colunas de data
    date_cols = [col for col in df_mapped.columns if 'data' in col.lower()]
    for col in date_cols:
        df_mapped[col] = pd.to_datetime(df_mapped[col], errors='coerce')
    
    # Garantir coluna data_de_abertura
    if 'data_de_abertura' not in df_mapped.columns:
        data_col = next((col for col in date_cols if col in df_mapped.columns), None)
        if data_col:
            df_mapped['data_de_abertura'] = df_mapped[data_col]
            logger.info(f"Usando {data_col} como data_de_abertura")
        else:
            logger.warning("Nenhuma coluna de data encontrada no PCM")
            df_mapped['data_de_abertura'] = pd.NaT
    
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


# =============================================================================
# FUNÇÕES PARA ANÁLISE DE FALHAS (RCA) E PLANOS DE AÇÃO
# =============================================================================

# Configuração de arquivos para RCA e Planos de Ação
FILE_CONFIG = {
    'analise_falhas': {
        'sheet_name': 0,
        'skiprows': 8,
        'column_map': {
            '% concluída': 'progresso',
            'Atribuída a': 'responsavel',
            'Nome': 'tarefa'
        }
    },
    'plano_acao': {
        'sheet_name': 0,
        'skiprows': 0,
        'column_map': {
            'Status da Ação': 'status',
            'Responsável pela Execução': 'responsavel',
            'Término planejado': 'prazo'
        }
    }
}


def _load_and_process_file(uploaded_file, config, file_type_name):
    """
    Função genérica para carregar e processar arquivo Excel (RCA/Planos).
    
    Args:
        uploaded_file: Arquivo Excel (Streamlit UploadedFile ou path)
        config: Configuração com sheet_name, skiprows, column_map
        file_type_name: Nome do tipo de arquivo (para mensagens)
        
    Returns:
        DataFrame processado ou vazio se erro
    """
    if not uploaded_file:
        return pd.DataFrame()
    
    try:
        # Ler Excel
        df = pd.read_excel(
            uploaded_file,
            sheet_name=config.get('sheet_name', 0),
            skiprows=config.get('skiprows', 0),
            header=0
        )
        
        logger.info(f"{file_type_name}: {len(df)} linhas, {len(df.columns)} colunas")
        
        # Padronizar colunas
        df_std = df.copy()
        df_std = standardize_columns(df_std)
        
        # Mapear colunas
        column_map = config['column_map']
        rename_map = {
            standardize_string(original_name): final_name
            for original_name, final_name in column_map.items()
        }
        
        df_std.rename(columns=rename_map, inplace=True)
        
        # Verificar colunas esperadas
        final_expected_cols = list(column_map.values())
        missing_cols = [col for col in final_expected_cols if col not in df_std.columns]
        
        if missing_cols:
            logger.error(f"{file_type_name}: Colunas essenciais não encontradas: {', '.join(missing_cols)}")
            return pd.DataFrame()
        
        logger.info(f"✓ {file_type_name} carregado com sucesso")
        return df_std
    
    except Exception as e:
        logger.error(f"Erro ao processar {file_type_name}: {e}")
        return pd.DataFrame()


def get_analise_df(uploaded_file):
    """
    Processa o arquivo de análise de falhas (RCA).
    
    Configuração:
    - skiprows=8 (pula cabeçalho)
    - Mapeia: '% concluída' → 'progresso'
              'Atribuída a' → 'responsavel'
              'Nome' → 'tarefa'
    
    Args:
        uploaded_file: Arquivo Excel de análise
        
    Returns:
        DataFrame com colunas: progresso, responsavel, tarefa
    """
    logger.info("Carregando arquivo de Análise de Falhas...")
    return _load_and_process_file(
        uploaded_file, 
        FILE_CONFIG['analise_falhas'], 
        "Análise de Falhas"
    )


def get_plano_df(uploaded_file):
    """
    Processa o arquivo de plano de ação.
    
    Configuração:
    - skiprows=0 (sem pular linhas)
    - Mapeia: 'Status da Ação' → 'status'
              'Responsável pela Execução' → 'responsavel'
              'Término planejado' → 'prazo'
    
    Args:
        uploaded_file: Arquivo Excel de plano de ação
        
    Returns:
        DataFrame com colunas: status, responsavel, prazo
    """
    logger.info("Carregando arquivo de Plano de Ação...")
    return _load_and_process_file(
        uploaded_file, 
        FILE_CONFIG['plano_acao'], 
        "Plano de Ação"
    )
