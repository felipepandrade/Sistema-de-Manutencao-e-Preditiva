"""
src/explainability/report_generator.py
Módulo de geração de relatórios (HTML/PDF).
"""

import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import logging
from datetime import datetime
import jinja2

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Gerador de relatórios executivos de manutenção preditiva.
    """
    
    def __init__(self, output_dir: Path = Path("outputs/reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Template HTML básico
        self.template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Manutenção Preditiva</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
                .card { background: #f9f9f9; padding: 20px; margin-bottom: 20px; border-radius: 5px; }
                .high-risk { color: #e74c3c; font-weight: bold; }
                .medium-risk { color: #f39c12; font-weight: bold; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .footer { margin-top: 50px; font-size: 0.8em; color: #7f8c8d; }
            </style>
        </head>
        <body>
            <h1>Relatório de Manutenção Preditiva</h1>
            <p><strong>Data de Geração:</strong> {{ date }}</p>
            <p><strong>Versão do Modelo:</strong> {{ model_version }}</p>
            
            <div class="card">
                <h2>Resumo Executivo</h2>
                <p>Total de Ativos Analisados: <strong>{{ total_assets }}</strong></p>
                <p>Ativos em Alto Risco (30 dias): <span class="high-risk">{{ high_risk_count }}</span></p>
                <p>Ativos em Médio Risco (30 dias): <span class="medium-risk">{{ medium_risk_count }}</span></p>
            </div>
            
            <h2>Top 10 Ativos Críticos</h2>
            <table>
                <thead>
                    <tr>
                        <th>Ativo</th>
                        <th>Probabilidade (30d)</th>
                        <th>Risco</th>
                        <th>Ação Recomendada</th>
                    </tr>
                </thead>
                <tbody>
                    {% for asset in top_assets %}
                    <tr>
                        <td>{{ asset.ativo }}</td>
                        <td>{{ "%.1f"|format(asset.prob * 100) }}%</td>
                        <td class="{{ 'high-risk' if asset.risco == 'Alto Risco' else 'medium-risk' }}">{{ asset.risco }}</td>
                        <td>{{ asset.acao }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            <div class="footer">
                Gerado automaticamente pelo Sistema Preditivo de Manutenção v2.0
            </div>
        </body>
        </html>
        """

    def generate_html_report(
        self,
        df_predictions: pd.DataFrame,
        model_version: str,
        filename: str = None
    ) -> Path:
        """
        Gera relatório HTML com base nas predições.
        """
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
        output_path = self.output_dir / filename
        
        # Preparar dados para o template
        total_assets = len(df_predictions)
        high_risk = df_predictions[df_predictions['Classe_30d'] == 'Alto Risco']
        medium_risk = df_predictions[df_predictions['Classe_30d'] == 'Médio Risco']
        
        # Top 10 críticos
        top_critical = pd.concat([high_risk, medium_risk]).sort_values(
            'ProbML_Media_30d', ascending=False
        ).head(10)
        
        top_assets_list = []
        for _, row in top_critical.iterrows():
            acao = "Inspeção Imediata" if row['Classe_30d'] == 'Alto Risco' else "Programar Inspeção"
            top_assets_list.append({
                'ativo': row['ativo_unico'],
                'prob': row['ProbML_Media_30d'],
                'risco': row['Classe_30d'],
                'acao': acao
            })
            
        # Renderizar template
        template = jinja2.Template(self.template_str)
        html_content = template.render(
            date=datetime.now().strftime('%d/%m/%Y %H:%M'),
            model_version=model_version,
            total_assets=total_assets,
            high_risk_count=len(high_risk),
            medium_risk_count=len(medium_risk),
            top_assets=top_assets_list
        )
        
        # Salvar arquivo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Relatório gerado em: {output_path}")
        return output_path
