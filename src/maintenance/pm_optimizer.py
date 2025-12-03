"""
src/maintenance/pm_optimizer.py
Módulo de otimização de manutenção preventiva usando IA Generativa (Gemini).
"""

import os
import google.generativeai as genai
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

class PMOptimizer:
    """
    Otimizador de planos de manutenção usando LLMs (Gemini).
    Gera recomendações personalizadas baseadas no risco e contexto do ativo.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """
        Inicializa o otimizador.
        
        Args:
            api_key: Chave de API do Google (se None, busca em env GEMINI_API_KEY)
            model_name: Nome do modelo Gemini a usar
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.model = None
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY não encontrada. Otimização via IA estará indisponível.")
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(model_name)
                logger.info(f"PM Optimizer inicializado com {model_name}")
            except Exception as e:
                logger.error(f"Erro ao configurar Gemini: {e}")
                self.model = None

    def generate_plan(
        self,
        asset_info: Dict[str, Any],
        prediction_info: Dict[str, Any],
        top_features: Dict[str, float]
    ) -> str:
        """
        Gera um plano de manutenção otimizado.
        
        Args:
            asset_info: Dict com dados do ativo (nome, instalação, etc.)
            prediction_info: Dict com probabilidade e risco
            top_features: Dict com features mais importantes e seus valores/impacto
            
        Returns:
            Texto com o plano sugerido
        """
        if not self.model:
            return "Erro: IA Generativa não configurada. Verifique a API Key."
        
        try:
            prompt = self._build_prompt(asset_info, prediction_info, top_features)
            
            logger.info(f"Solicitando otimização para {asset_info.get('ativo_unico', 'Ativo Desconhecido')}...")
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Erro na geração do plano: {e}")
            return f"Erro ao gerar plano: {str(e)}"

    def _build_prompt(
        self,
        asset_info: Dict[str, Any],
        prediction_info: Dict[str, Any],
        top_features: Dict[str, float]
    ) -> str:
        """Constrói o prompt para a LLM."""
        
        features_text = "\n".join([f"- {k}: {v:.2f}% impacto" for k, v in top_features.items()])
        
        prompt = f"""
        Atue como um Engenheiro Especialista em Manutenção de Gasodutos e Confiabilidade.
        
        Analise o seguinte cenário de risco preditivo e gere um Plano de Otimização de Manutenção Preventiva (PM).
        
        DADOS DO ATIVO:
        - Identificação: {asset_info.get('ativo_unico')}
        - Instalação: {asset_info.get('instalacao')}
        - Módulo: {asset_info.get('modulo_envolvido')}
        - Tipo: {asset_info.get('ativo')}
        
        PREDIÇÃO DE FALHA (Modelo ML):
        - Horizonte: {prediction_info.get('horizonte')} dias
        - Probabilidade de Falha: {prediction_info.get('probabilidade', 0):.1%}
        - Classificação de Risco: {prediction_info.get('risco')}
        
        FATORES CONTRIBUINTES (Top Features):
        {features_text}
        
        TAREFA:
        Gere um relatório técnico conciso contendo:
        1. DIAGNÓSTICO: Interpretação técnica do porquê o risco está alto/médio, baseado nas features.
        2. AÇÃO RECOMENDADA: O que deve ser feito imediatamente e o que pode ser programado.
        3. RECURSOS: Ferramentas, peças ou especialidades necessárias.
        4. JUSTIFICATIVA: Por que intervir agora (ROI/Segurança).
        
        Use linguagem técnica apropriada para o setor de Óleo e Gás. Seja direto e acionável.
        """
        return prompt
