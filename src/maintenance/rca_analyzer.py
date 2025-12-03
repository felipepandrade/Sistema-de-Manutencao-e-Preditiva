"""
src/maintenance/rca_analyzer.py
Módulo de Análise de Causa Raiz (Root Cause Analysis) - Versão Aprimorada.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from collections import Counter

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RCAAnalyzer:
    """
    Analisador de Causa Raiz para eventos de falha.
    
    Funcionalidades:
    - Identifica anomalias em features que precederam a falha
    - Analisa padrões históricos de falhas similares
    - Ranqueia causas prováveis
    - Gera recomendações de ações corretivas
    """
    
    def __init__(self, zscore_threshold: float = 2.0):
        """
        Args:
            zscore_threshold: Limiar de Z-score para considerar anomalia (padrão: 2.0)
        """
        self.zscore_threshold = zscore_threshold
        self.failure_patterns_db = []  # Banco de padrões de falhas conhecidas

    def analyze(
        self,
        asset_history: pd.DataFrame,
        failure_date: pd.Timestamp,
        feature_names: List[str],
        window_days: int = 7,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Analisa dados históricos precedentes à falha para identificar desvios.
        
        Args:
            asset_history: Histórico completo do ativo
            failure_date: Data da falha a investigar
            feature_names: Features a analisar
            window_days: Janela de análise antes da falha (padrão: 7 dias)
            include_recommendations: Se True, gera recomendações de ação
            
        Returns:
            Relatório de RCA com causas prováveis e recomendações
        """
        logger.info(f"Analisando RCA para falha em {failure_date}")
        
        # 1. Validar dados
        if asset_history.empty:
            return {"error": "Histórico vazio"}
            
        ativo_unico = asset_history['ativo_unico'].iloc[0] if 'ativo_unico' in asset_history.columns else "Unknown"
        
        # 2. Filtrar janela antes da falha
        start_date = failure_date - pd.Timedelta(days=window_days)
        df_window = asset_history[
            (asset_history['data'] >= start_date) & 
            (asset_history['data'] < failure_date)
        ].copy()
        
        if df_window.empty:
            return {
                "ativo_unico": ativo_unico,
                "error": f"Sem dados na janela de {window_days} dias antes da falha"
            }
        
        # 3. Análise de desvios (anomalias)
        anomalies = self._detect_anomalies(
            asset_history,
            df_window,
            start_date,
            feature_names
        )
        
        # 4. Análise de padrões históricos
        similar_failures = self._find_similar_failures(
            asset_history,
            failure_date,
            anomalies,
            feature_names
        )
        
        # 5. Ranquear causas p

róveis
        root_causes = self._rank_root_causes(anomalies, similar_failures)
        
        # 6. Gerar recomendações
        recommendations = []
        if include_recommendations:
            recommendations = self._generate_recommendations(root_causes, ativo_unico)
        
        # 7. Compilar relatório
        report = {
            "ativo_unico": ativo_unico,
            "failure_date": failure_date.isoformat(),
            "analyzed_window_days": window_days,
            "anomalies_detected": len(anomalies),
            "top_root_causes": root_causes[:5],
            "similar_historical_failures": len(similar_failures),
            "recommendations": recommendations,
            "confidence_score": self._calculate_confidence(anomalies, similar_failures)
        }
        
        logger.info(f"RCA concluída: {len(anomalies)} anomalias detectadas, "
                   f"{len(similar_failures)} falhas similares encontradas")
        
        return report

    def _detect_anomalies(
        self,
        asset_history: pd.DataFrame,
        df_window: pd.DataFrame,
        start_date: pd.Timestamp,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Detecta anomalias estatísticas nas features."""
        # Baseline: média histórica ANTES da janela de falha
        df_baseline = asset_history[asset_history['data'] < start_date]
        
        if df_baseline.empty:
            # Fallback: usar a própria janela (menos ideal)
            baseline_stats = df_window[feature_names].mean()
            baseline_std = df_window[feature_names].std()
            logger.warning("Baseline calculado com dados da janela de falha (histórico insuficiente)")
        else:
            baseline_stats = df_baseline[feature_names].mean()
            baseline_std = df_baseline[feature_names].std()
        
        # Estatísticas na janela de falha
        window_mean = df_window[feature_names].mean()
        window_std = df_window[feature_names].std()
        
        anomalies = []
        
        for feature in feature_names:
            std = baseline_std[feature]
            
            if std == 0 or pd.isna(std):
                continue
                
            # Z-score
            z_score = (window_mean[feature] - baseline_stats[feature]) / std
            
            if abs(z_score) > self.zscore_threshold:
                # Calcular tendência (crescente/decrescente)
                if len(df_window) > 1:
                    trend = "crescente" if df_window[feature].iloc[-1] > df_window[feature].iloc[0] else "decrescente"
                else:
                    trend = "estável"
                
                severity = self._classify_severity(abs(z_score))
                
                anomalies.append({
                    "feature": feature,
                    "z_score": float(z_score),
                    "severity": severity,
                    "window_mean": float(window_mean[feature]),
                    "baseline_mean": float(baseline_stats[feature]),
                    "deviation_pct": float(((window_mean[feature] - baseline_stats[feature]) / baseline_stats[feature]) * 100) if baseline_stats[feature] != 0 else 0,
                    "impact": "Aumento" if z_score > 0 else "Queda",
                    "trend": trend,
                    "volatility": float(window_std[feature]) if pd.notna(window_std[feature]) else 0.0
                })
        
        # Ordenar por magnitude (Z-score absoluto)
        anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        return anomalies

    def _classify_severity(self, abs_zscore: float) -> str:
        """Classifica severidade da anomalia."""
        if abs_zscore >= 4.0:
            return "CRÍTICA"
        elif abs_zscore >= 3.0:
            return "ALTA"
        elif abs_zscore >= 2.0:
            return "MODERADA"
        else:
            return "BAIXA"

    def _find_similar_failures(
        self,
        asset_history: pd.DataFrame,
        current_failure_date: pd.Timestamp,
        current_anomalies: List[Dict],
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Busca falhas históricas com padrões similares de anomalias.
        """
        similar_failures = []
        
        # Features com anomalias significativas
        anomalous_features = {a['feature'] for a in current_anomalies}
        
        if not anomalous_features:
            return []
        
        # Agrupar por data (cada data é um evento de falha)
        failure_dates = asset_history['data'].drop_duplicates().sort_values()
        
        for past_failure_date in failure_dates:
            # Pular a falha atual
            if past_failure_date >= current_failure_date:
                continue
            
            # Janela antes da falha passada
            start_past = past_failure_date - pd.Timedelta(days=7)
            df_past_window = asset_history[
                (asset_history['data'] >= start_past) & 
                (asset_history['data'] < past_failure_date)
            ]
            
            if df_past_window.empty:
                continue
            
            # Verificar se features similares estavam anômalas
            past_anomalous_features = set()
            
            for feature in anomalous_features:
                if feature not in df_past_window.columns:
                    continue
                    
                # Calcular z-score simplificado
                historical_mean = asset_history[asset_history['data'] < start_past][feature].mean()
                historical_std = asset_history[asset_history['data'] < start_past][feature].std()
                
                if historical_std == 0 or pd.isna(historical_std):
                    continue
                
                past_window_mean = df_past_window[feature].mean()
                z = abs((past_window_mean - historical_mean) / historical_std)
                
                if z > self.zscore_threshold:
                    past_anomalous_features.add(feature)
            
            # Similaridade (Jaccard)
            if past_anomalous_features:
                similarity = len(anomalous_features & past_anomalous_features) / len(anomalous_features | past_anomalous_features)
                
                if similarity >= 0.3:  # Pelo menos 30% de overlap
                    similar_failures.append({
                        "date": past_failure_date.isoformat(),
                        "similarity_score": float(similarity),
                        "common_anomalies": list(anomalous_features & past_anomalous_features),
                        "days_ago": (current_failure_date - past_failure_date).days
                    })
        
        # Ordenar por similaridade
        similar_failures.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_failures[:5]  # Top 5 mais similares

    def _rank_root_causes(
        self,
        anomalies: List[Dict],
        similar_failures: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Ranqueia causas prováveis baseado em:
        1. Magnitude da anomalia (Z-score)
        2. Recorrência em falhas similares
        3. Severidade classificada
        """
        # Contar frequência de features em falhas similares
        feature_frequency = Counter()
        for failure in similar_failures:
            for feat in failure['common_anomalies']:
                feature_frequency[feat] += 1
        
        root_causes = []
        
        for anomaly in anomalies:
            feature = anomaly['feature']
            
            # Score baseado em z-score + frequência histórica
            recurrence_score = feature_frequency.get(feature, 0) / max(len(similar_failures), 1)
            zscore_score = min(abs(anomaly['z_score']) / 4.0, 1.0)  # Normalizado 0-1
            
            # Score final (média ponderada)
            combined_score = (zscore_score * 0.7) + (recurrence_score * 0.3)
            
            # Confiança baseada em evidências
            confidence = "ALTA" if recurrence_score > 0.5 and zscore_score > 0.75 else \
                        "MÉDIA" if recurrence_score > 0.3 or zscore_score > 0.5 else \
                        "BAIXA"
            
            root_causes.append({
                "cause_feature": feature,
                "probability_score": float(combined_score),
                "confidence": confidence,
                "z_score": anomaly['z_score'],
                "deviation_pct": anomaly['deviation_pct'],
                "historical_recurrence": int(feature_frequency.get(feature, 0)),
                "severity": anomaly['severity'],
                "description": self._generate_cause_description(anomaly)
            })
        
        # Ordenar por score
        root_causes.sort(key=lambda x: x['probability_score'], reverse=True)
        
        return root_causes

    def _generate_cause_description(self, anomaly: Dict) -> str:
        """Gera descrição humanizada da anomalia."""
        feature = anomaly['feature']
        impact = anomaly['impact']
        deviation = abs(anomaly['deviation_pct'])
        
        description = f"{feature.replace('_', ' ').title()}: {impact.lower()} de {deviation:.1f}% em relação ao baseline"
        
        if anomaly.get('trend') == 'crescente':
            description += " (tendência crescente)"
        elif anomaly.get('trend') == 'decrescente':
            description += " (tendência decrescente)"
        
        return description

    def _generate_recommendations(
        self,
        root_causes: List[Dict],
        ativo_unico: str
    ) -> List[Dict[str, str]]:
        """
        Gera recomendações de ações corretivas baseadas nas causas raiz.
        """
        recommendations = []
        
        # Mapeamento de features para ações (baseado em conhecimento de domínio)
        action_map = {
            "tbf": "Reduzir intervalo de manutenção preventiva - TBF está decaindo",
            "tbf_mean": "Intensificar monitoramento - Média de TBF abaixo do esperado",
            "tbf_volatilidade": "Investigar variações operacionais - Alta volatilidade detectada",
            "falhas_acumuladas": "Ativo crítico - Considerar substituição ou overhaul",
            "idade_ativo": "Ativo envelhecido - Planejar substituição ou recondicionamento",
            "temperatura": "Verificar sistema de resfriamento e lubrificação",
            "pressao": "Inspecionar vedações e sistema de pressurização",
            "vibracao": "Verificar alinhamento, balanceamento e fixação",
            "corrosao": "Inspecionar materiais e proteção anticorrosiva"
        }
        
        for i, cause in enumerate(root_causes[:3], 1):  # Top 3 causas
            feature = cause['cause_feature']
            
            # Buscar ação específica ou ação genérica
            action = None
            for key, desc in action_map.items():
                if key in feature.lower():
                    action = desc
                    break
            
            if not action:
                action = f"Analisar comportamento de {feature.replace('_', ' ')} - Desvio de {cause['deviation_pct']:.1f}%"
            
            recommendations.append({
                "priority": i,
                "action": action,
                "cause": cause['cause_feature'],
                "confidence": cause['confidence'],
                "urgency": "ALTA" if cause['severity'] in ["CRÍTICA", "ALTA"] else "MÉDIA"
            })
        
        # Recomendação geral
        if len(root_causes) > 0:
            recommendations.append({
                "priority": 99,
                "action": f"Revisar logs operacionais de {ativo_unico} no período precedente à falha",
                "cause": "análise_geral",
                "confidence": "ALTA",
                "urgency": "MÉDIA"
            })
        
        return recommendations

    def _calculate_confidence(
        self,
        anomalies: List[Dict],
        similar_failures: List[Dict]
    ) -> float:
        """
        Calcula score de confiança da análise RCA.
        
        Confiança é alta quando:
        - Múltiplas anomalias significativas detectadas
        - Falhas similares históricas encontradas
        """
        if not anomalies:
            return 0.0
        
        # Score de anomalias (maior z-score = maior confiança)
        avg_zscore = np.mean([abs(a['z_score']) for a in anomalies])
        anomaly_score = min(avg_zscore / 4.0, 1.0)
        
        # Score de evidências históricas
        history_score = min(len(similar_failures) / 3.0, 1.0)
        
        # Confiança combinada
        confidence = (anomaly_score * 0.6) + (history_score * 0.4)
        
        return float(confidence)

    def analyze_with_ai(
        self,
        rca_report: Dict[str, Any],
        failure_descriptions: Optional[List[str]] = None,
        ai_provider: str = "gemini"
    ) -> Dict[str, Any]:
        """
        Enriquece análise RCA com IA Generativa para análise de texto.
        
        Args:
            rca_report: Relatório RCA gerado pelo método analyze()
            failure_descriptions: Descrições textuais de falhas (opcional)
            ai_provider: Provedor de IA ("gemini" ou "openai")
            
        Returns:
            Relatório enriquecido com insights de IA
        """
        logger.info("Iniciando análise RCA com IA Generativa...")
        
        try:
            # Construir contexto para IA
            context = self._build_ai_context(rca_report, failure_descriptions)
            
            # Gerar análise com IA
            ai_analysis = self._query_ai(context, ai_provider)
            
            # Integrar análise de IA ao relatório
            enhanced_report = rca_report.copy()
            enhanced_report['ai_analysis'] = {
                'provider': ai_provider,
                'insights': ai_analysis.get('insights', ''),
                'additional_recommendations': ai_analysis.get('recommendations', []),
                'risk_assessment': ai_analysis.get('risk_assessment', ''),
                'predicted_root_cause': ai_analysis.get('predicted_root_cause', '')
            }
            
            # Salvar contexto para chat conversacional
            enhanced_report['chat_context'] = context
            
            logger.info("✓ Análise com IA concluída")
            
            return enhanced_report
            
        except Exception as e:
            logger.error(f"Erro na análise com IA: {e}")
            rca_report['ai_analysis'] = {'error': str(e)}
            return rca_report

    def _build_ai_context(
        self,
        rca_report: Dict[str, Any],
        failure_descriptions: Optional[List[str]] = None
    ) -> str:
        """Constrói contexto estruturado para IA."""
        context_parts = [
            "# Análise de Causa Raiz (RCA)",
            f"\n**Ativo:** {rca_report.get('ativo_unico', 'N/A')}",
            f"**Data da Falha:** {rca_report.get('failure_date', 'N/A')}",
            f"**Janela de Análise:** {rca_report.get('analyzed_window_days', 0)} dias",
            f"**Anomalias Detectadas:** {rca_report.get('anomalies_detected', 0)}",
            f"**Confiança da Análise:** {rca_report.get('confidence_score', 0):.2f}",
            "\n## Top Causas Raiz Identificadas:\n"
        ]
        
        # Adicionar causas raiz
        for i, cause in enumerate(rca_report.get('top_root_causes', [])[:3], 1):
            context_parts.append(
                f"{i}. **{cause['cause_feature']}**\n"
                f"   - Score: {cause['probability_score']:.2f}\n"
                f"   - Confiança: {cause['confidence']}\n"
                f"   - Desvio: {cause['deviation_pct']:.1f}%\n"
                f"   - Recorrência histórica: {cause['historical_recurrence']} vezes\n"
                f"   - Descrição: {cause['description']}\n"
            )
        
        # Adicionar falhas similares
        similar = rca_report.get('similar_historical_failures', 0)
        if similar > 0:
            context_parts.append(f"\n**Falhas Históricas Similares:** {similar} encontradas")
        
        # Adicionar descrições textuais se fornecidas
        if failure_descriptions:
            context_parts.append("\n## Descrições de Falhas Relacionadas:\n")
            for desc in failure_descriptions[:5]:  # Limitar a 5
                context_parts.append(f"- {desc}\n")
        
        return "\n".join(context_parts)

    def _query_ai(self, context: str, provider: str = "gemini") -> Dict[str, Any]:
        """Query IA Generativa com contexto da análise."""
        prompt = f"""Você é um especialista em análise de causa raiz (RCA) de equipamentos industriais.

Analise o seguinte relatório RCA e forneça:

1. **Insights Adicionais:** Interpretação dos dados e padrões identificados
2. **Causa Raiz Provável:** Sua melhor hipótese sobre a causa principal
3. **Avaliação de Risco:** Severidade e probabilidade de recorrência
4. **Recomendações Complementares:** Ações específicas além das já sugeridas

{context}

Responda em formato JSON com as chaves:
- insights (string)
- predicted_root_cause (string)
- risk_assessment (string)
- recommendations (list of strings)
"""
        
        if provider == "gemini":
            return self._query_gemini(prompt)
        elif provider == "openai":
            return self._query_openai(prompt)
        else:
            raise ValueError(f"Provedor não suportado: {provider}")

    def _query_gemini(self, prompt: str) -> Dict[str, Any]:
        """Query Google Gemini."""
        try:
            import google.generativeai as genai
            import os
            import json
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY não configurada")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            response = model.generate_content(prompt)
            
            # Parsear JSON da resposta
            try:
                # Tentar extrair JSON da resposta
                text = response.text
                # Remover markdown code blocks se existirem
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                
                result = json.loads(text.strip())
                return result
            except:
                # Fallback: retornar texto bruto
                return {
                    "insights": response.text,
                    "predicted_root_cause": "Análise disponível no campo insights",
                    "risk_assessment": "Verificar insights",
                    "recommendations": []
                }
                
        except ImportError:
            raise ImportError("google-generativeai não instalado. Execute: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Erro ao consultar Gemini: {e}")
            raise

    def _query_openai(self, prompt: str) -> Dict[str, Any]:
        """Query OpenAI GPT."""
        try:
            import openai
            import os
            import json
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY não configurada")
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "Você é um especialista em RCA de equipamentos industriais. Responda sempre em JSON válido."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except ImportError:
            raise ImportError("openai não instalado. Execute: pip install openai")
        except Exception as e:
            logger.error(f"Erro ao consultar OpenAI: {e}")
            raise

    def chat_with_context(
        self,
        rca_report: Dict[str, Any],
        user_question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        ai_provider: str = "gemini"
    ) -> Dict[str, str]:
        """
        Chat conversacional com contexto da análise RCA.
        
        Args:
            rca_report: Relatório RCA (deve conter 'chat_context')
            user_question: Pergunta do usuário
            chat_history: Histórico de conversação (opcional)
            ai_provider: Provedor de IA
            
        Returns:
            Dict com 'question' e 'answer'
        """
        logger.info(f"Chat RCA: {user_question[:50]}...")
        
        # Recuperar contexto base
        base_context = rca_report.get('chat_context', '')
        
        if not base_context:
            base_context = self._build_ai_context(rca_report, None)
        
        # Construir prompt com histórico
        messages = [
            {
                "role": "system",
                "content": f"""Você é um especialista em análise de causa raiz (RCA) de equipamentos industriais.

Você está analisando o seguinte caso:

{base_context}

Responda perguntas do usuário de forma precisa, baseando-se nos dados do relatório RCA acima.
Se a pergunta não puder ser respondida com os dados disponíveis, seja honesto e sugira análises adicionais."""
            }
        ]
        
        # Adicionar histórico
        if chat_history:
            for entry in chat_history[-5:]:  # Últimas 5 mensagens
                messages.append({"role": "user", "content": entry.get("question", "")})
                messages.append({"role": "assistant", "content": entry.get("answer", "")})
        
        # Adicionar pergunta atual
        messages.append({"role": "user", "content": user_question})
        
        # Query IA
        try:
            if ai_provider == "gemini":
                answer = self._chat_gemini(messages)
            elif ai_provider == "openai":
                answer = self._chat_openai(messages)
            else:
                raise ValueError(f"Provedor não suportado: {ai_provider}")
            
            return {
                "question": user_question,
                "answer": answer
            }
            
        except Exception as e:
            logger.error(f"Erro no chat: {e}")
            return {
                "question": user_question,
                "answer": f"Erro ao processar pergunta: {str(e)}"
            }

    def _chat_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Chat com Gemini."""
        import google.generativeai as genai
        import os
        
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Gemini não usa formato de mensagens diretamente, concatenar contexto
        full_prompt = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        response = model.generate_content(full_prompt)
        return response.text

    def _chat_openai(self, messages: List[Dict[str, str]]) -> str:
        """Chat com OpenAI."""
        import openai
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages
        )
        
        return response.choices[0].message.content

