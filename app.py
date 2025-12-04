"""
app.py
Aplicação Streamlit - Sistema Preditivo de Manutenção v2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import yaml
import subprocess
import time

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logging_config import load_config, get_logger
from src.data.loaders import load_falhas_excel
from src.features.engineering import FeatureEngineeringPipeline
from src.features.target_builder import create_multi_horizon_targets
from src.inference.predictor import PredictorPipeline

# Configuração da página
st.set_page_config(
    page_title="Sistema Preditivo de Manutenção",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregar configuração
@st.cache_resource
def load_app_config():
    """Carrega configuração do sistema."""
    config_path = Path(__file__).parent / "config.yaml"
    return load_config(config_path)

config = load_app_config()

# Título principal
st.title("🔧 Sistema Preditivo de Manutenção")
st.markdown(f"**Versão {config['project']['version']}** | Análise Preditiva de Falhas em Equipamentos")

# Sidebar
st.sidebar.header("📁 Carregar Dados")

# Inicializar session_state para os DataFrames
if 'falhas_df' not in st.session_state:
    st.session_state.falhas_df = pd.DataFrame()
if 'pcm_df' not in st.session_state:
    st.session_state.pcm_df = pd.DataFrame()
if 'analise_df' not in st.session_state:
    st.session_state.analise_df = pd.DataFrame()
if 'plano_df' not in st.session_state:
    st.session_state.plano_df = pd.DataFrame()

st.sidebar.markdown("Faça o upload dos arquivos de dados para análise.")

# 1. Upload de Ocorrências (Falhas) - OBRIGATÓRIO
uploaded_file = st.sidebar.file_uploader(
    "1. Ocorrências (Falhas)",
    type=['xlsx', 'xls'],
    help="Arquivo deve conter histórico de falhas dos equipamentos",
    key='upload_falhas'
)

# 2. Upload de PCM (Ordens de Serviço) - OPCIONAL
uploaded_pcm = st.sidebar.file_uploader(
    "2. Ordens de Serviço (PCM)",
    type=['xlsx'],
    help="Arquivo de Plano de Controle de Manutenção",
    key='upload_pcm'
)

# 3. Upload de Análise de Falhas (RCA) - OPCIONAL
uploaded_analise = st.sidebar.file_uploader(
    "3. Análise de Falhas (RCA)",
    type=['xlsx'],
    help="Arquivo de análise de causa raiz",
    key='upload_analise'
)

# 4. Upload de Planos de Ação - OPCIONAL
uploaded_plano = st.sidebar.file_uploader(
    "4. Planos de Ação",
    type=['xlsx'],
    help="Arquivo de planos de ação para falhas",
    key='upload_plano'
)

st.sidebar.divider()

# Processar arquivos carregados
from src.data import load_pcm_excel, get_analise_df, get_plano_df

if uploaded_pcm:
    try:
        st.session_state.pcm_df = load_pcm_excel(uploaded_pcm)
        if not st.session_state.pcm_df.empty:
            st.sidebar.success("✓ PCM carregado")
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar PCM: {e}")

if uploaded_analise:
    try:
        st.session_state.analise_df = get_analise_df(uploaded_analise)
        if not st.session_state.analise_df.empty:
            st.sidebar.success("✓ Análise de Falhas carregada")
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar Análise: {e}")

if uploaded_plano:
    try:
        st.session_state.plano_df = get_plano_df(uploaded_plano)
        if not st.session_state.plano_df.empty:
            st.sidebar.success("✓ Plano de Ação carregado")
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar Plano: {e}")


# Opções
st.sidebar.header("⚙️ Configurações")

use_cached_model = st.sidebar.checkbox(
    "Usar modelo treinado",
    value=True,
    help="Carrega modelo pré-treinado para inferência"
)

show_debug_info = st.sidebar.expander("🐛 Debug Info")

# Main content
if uploaded_file is None:
    st.info("👈 Faça upload de um arquivo Excel na barra lateral para começar")
    
    # Instruções
    st.markdown("### 📖 Como Usar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Preparar Dados**
        - Histórico de falhas em Excel
        - Colunas essenciais:
          - Data da falha
          - Equipamento
          - Instalação
          - Módulo
        
        **2. Upload**
        - Clique em "Browse files"
        - Selecione o arquivo Excel
        """)
    
    with col2:
        st.markdown("""
        **3. Análise**
        - Sistema gera features automáticas
        - Prediz falhas para 3, 7, 15 e 30 dias
        - Classifica risco (Alto/Médio/Baixo)
        
        **4. Ações**
        - Visualize predições
        - Exporte resultados
        - Otimize planos de manutenção
        """)
    
    st.markdown("### 📊 Exemplo de Resultados")
    
    # Exemplo de dados
    exemplo_df = pd.DataFrame({
        'Ativo': ['COMP-01', 'COMP-02', 'VÁLV-03', 'COMP-04'],
        'Prob 7d': [0.15, 0.42, 0.78, 0.25],
        'Risco 7d': ['Baixo', 'Médio', 'Alto', 'Baixo'],
        'Prob 30d': [0.35, 0.68, 0.92, 0.51],
        'Risco 30d': ['Médio', 'Médio', 'Alto', 'Médio']
    })
    
    st.dataframe(exemplo_df, use_container_width=True)

else:
    # Processar dados
    try:
        # Carregar dados (validação é feita dentro de load_falhas_excel)
        with st.spinner("📥 Carregando dados..."):
            df_raw = load_falhas_excel(uploaded_file, config)
        
        # Verificar se carregamento foi bem-sucedido
        if df_raw.empty:
            st.error("❌ Erro ao carregar arquivo. Verifique os logs no terminal para detalhes.")
            st.stop()
        
        st.success(f"✓ Dados carregados: {len(df_raw)} registros, {df_raw['ativo_unico'].nunique()} ativos únicos")
        
        if show_debug_info:
            with show_debug_info:
                st.write("**Dados brutos:**")
                st.dataframe(df_raw.head(), use_container_width=True)
                st.write(f"Colunas: {list(df_raw.columns)}")
        
        # Feature Engineering (com cache)
        @st.cache_data(show_spinner=False)
        def generate_features(df, feature_config):
            pipeline = FeatureEngineeringPipeline(
                lags=feature_config['lags'],
                rolling_windows=feature_config['rolling_windows'],
                include_sazonalidade=feature_config['include_sazonalidade'],
                include_interacoes=feature_config['include_interacoes']
            )
            return pipeline.fit_transform(df), pipeline.get_feature_names()
        
        with st.spinner("🔧 Gerando features..."):
            df_features, feature_names = generate_features(df_raw, config['features'])
        
        st.success(f"✓ Features criadas: {len(feature_names)}")
        
        if show_debug_info:
            with show_debug_info:
                st.write("**Features geradas:**")
                st.write(feature_names[:10])  # Primeiras 10
        
        # Verificar se existem modelos treinados
        models_dir = Path(config['paths']['models'])
        latest_path = models_dir / 'latest'
        models_exist = latest_path.exists() and len(list(latest_path.glob('*.pkl'))) > 0
        
        if not models_exist:
            st.info("ℹ️ Nenhum modelo treinado encontrado.")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🚀 Treinar Modelos Agora", type="primary"):
                    with st.spinner("Treinando modelos... Isso pode levar alguns minutos."):
                        import subprocess
                        import sys
                        import tempfile
                        
                        try:
                            # Salvar df_raw temporariamente como Excel para o cli_train usar
                            temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False)
                            df_raw.to_excel(temp_file.name, index=False)
                            temp_file.close()
                            
                            # Executar cli_train.py com argumento --data
                            result = subprocess.run(
                                [sys.executable, "cli_train.py", "--data", temp_file.name],
                                capture_output=True,
                                text=True,
                                cwd=Path(__file__).parent
                            )
                            
                            # Limpar arquivo temporário
                            Path(temp_file.name).unlink(missing_ok=True)
                            
                            if result.returncode == 0:
                                st.success("✓ Treinamento concluído com sucesso!")
                                st.rerun()  # Recarregar para usar novos modelos
                            else:
                                st.error(f"❌ Erro no treinamento: {result.stderr}")
                                st.code(result.stdout)
                        except Exception as e:
                            st.error(f"❌ Erro ao executar treinamento: {e}")
            
            with col2:
                st.markdown("Ou vá para a aba **🧠 Treinamento** para mais opções.")
        
        # Tabs principais
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Predições", 
            "🔍 Explicabilidade", 
            "🛠️ Otimização PM", 
            "📈 Análise Exploratória",
            "🧠 Treinamento",
            "📄 Relatórios",
            "⚙️ Configurações",
            "🔬 Análise RCA"
        ])

        
        with tab1:
            st.header("Predições de Falhas")
            
            if not models_exist:
                st.warning("⚠️ Nenhum modelo treinado encontrado. Execute o treinamento na aba '🧠 Treinamento' ou clique no botão acima.")
            elif use_cached_model:
                # Carregar modelo e predizer
                with st.spinner("🤖 Gerando predições..."):
                    try:
                        predictor = PredictorPipeline(model_version='latest', config=config)
                        
                        if not predictor.models:
                            st.error("❌ Erro ao carregar modelos. Verifique os logs.")
                            st.info("💡 Tente retreinar os modelos na aba '🧠 Treinamento'.")
                        else:
                            df_predictions = predictor.predict(df_features)
                            st.success(f"✓ Predições geradas para {len(df_predictions)} ativos")
                            
                            # Guardar no session_state para outras abas
                            st.session_state['df_predictions'] = df_predictions
                    
                    except Exception as e:
                        st.error(f"❌ Erro ao gerar predições: {e}")
                        st.info("💡 Tente retreinar os modelos na aba '🧠 Treinamento'.")
                        df_predictions = None
                
                # Continuar apenas se predições foram geradas
                if 'df_predictions' in st.session_state and st.session_state['df_predictions'] is not None:
                    df_predictions = st.session_state['df_predictions']
                    
                    # Filtros
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        horizonte_filter = st.selectbox(
                            "Horizonte de Análise",
                            options=[3, 7, 15, 30],
                            index=1,  # 7 dias padrão
                            format_func=lambda x: f"{x} dias"
                        )
                    
                    with col2:
                        risco_filter = st.multiselect(
                            "Filtrar por Risco",
                            options=['Alto Risco', 'Médio Risco', 'Baixo Risco'],
                            default=['Alto Risco', 'Médio Risco']
                        )
                    
                    with col3:
                        n_show = st.slider(
                            "Quantidade a exibir",
                            min_value=10,
                            max_value=100,
                            value=20,
                            step=10
                        )
                    
                    # Filtrar dados
                    classe_col = f'Classe_{horizonte_filter}d'
                    prob_col = f'ProbML_Media_{horizonte_filter}d'
                    
                    df_filtered = df_predictions[df_predictions[classe_col].isin(risco_filter)].copy()
                    df_filtered = df_filtered.sort_values(prob_col, ascending=False).head(n_show)
                    
                    # Gráfico de barras
                    fig = go.Figure()
                    
                    colors = []
                    for risco in df_filtered[classe_col]:
                        if risco == 'Alto Risco':
                            colors.append(config['ui']['colors']['alto_risco'])
                        elif risco == 'Médio Risco':
                            colors.append(config['ui']['colors']['medio_risco'])
                        else:
                            colors.append(config['ui']['colors']['baixo_risco'])
                    
                    fig.add_trace(go.Bar(
                        x=df_filtered['ativo_unico'],
                        y=df_filtered[prob_col],
                        marker_color=colors,
                        text=[f"{p:.1%}" for p in df_filtered[prob_col]],
                        textposition='outside',
                        name='Probabilidade de Falha'
                    ))
                    
                    fig.update_layout(
                        title=f"Top {n_show} Ativos - Risco de Falha ({horizonte_filter} dias)",
                        xaxis_title="Ativo",
                        yaxis_title="Probabilidade de Falha",
                        yaxis=dict(tickformat='.0%'),
                        height=500,
                        showlegend=False,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabela detalhada
                    st.subheader("Detalhamento das Predições")
                    
                    # Preparar colunas para exibição
                    display_cols = ['ativo_unico']
                    for h in [3, 7, 15, 30]:
                        display_cols.extend([f'ProbML_Media_{h}d', f'Classe_{h}d'])
                    display_cols.append('ModeloEst')
                    
                    # Filtrar colunas que existem
                    display_cols = [col for col in display_cols if col in df_filtered.columns]
                    
                    df_display = df_filtered[display_cols].copy()
                    
                    # Formatar probabilidades
                    for col in df_display.columns:
                        if 'ProbML_Media' in col:
                            df_display[col] = df_display[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                    
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Exportar
                    st.subheader("Exportar Resultados")
                    csv = df_predictions.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Download CSV Completo",
                        data=csv,
                        file_name=f"predicoes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Guardar predições no session_state
                    st.session_state['df_predictions'] = df_predictions
            
            else:
                st.info("Marque 'Usar modelo treinado' na barra lateral para gerar predições")

        with tab2:
            st.header("🔍 Explicabilidade (SHAP) [2/7]")
            
            if 'df_predictions' not in st.session_state:
                st.warning("⚠️ **Ação Necessária:** Vá para a aba '📊 Predições' e gere predições primeiro.")
                st.info("💡 Esta aba depende de predições geradas para funcionar.")
            else:
                from src.explainability.shap_explainer import SHAPExplainer
                from src.utils.io import load_model_artifacts
                
                st.markdown("Entenda quais fatores mais contribuíram para o risco de falha.")
                
                # Selecionar ativo
                ativos_risco = st.session_state['df_predictions'][
                    st.session_state['df_predictions']['Classe_30d'].isin(['Alto Risco', 'Médio Risco'])
                ]['ativo_unico'].unique()
                
                if len(ativos_risco) == 0:
                    st.warning("Nenhum ativo de Alto/Médio risco encontrado para análise.")
                else:
                    selected_asset = st.selectbox("Selecione um ativo para analisar:", options=ativos_risco)
                    
                    if st.button("Calcular Importância de Features"):
                        with st.spinner("Carregando modelo e calculando SHAP values..."):
                            try:
                                # Carregar modelo (usando 30d como referência)
                                models_dir = Path(config['paths']['models'])
                                latest_path = models_dir / 'latest'
                                artifacts = load_model_artifacts(latest_path, horizontes=[30])
                                model = artifacts['models'][30]['model']
                                feature_names = artifacts['feature_names']
                                
                                # Pegar dados do ativo
                                asset_data = df_features[df_features['ativo_unico'] == selected_asset].iloc[[-1]]
                                X_sample = asset_data[feature_names]
                                
                                # Explicar
                                explainer = SHAPExplainer(model, feature_names=feature_names)
                                shap_values = explainer.explain(X_sample)
                                
                                # Plot Waterfall
                                st.subheader(f"Por que {selected_asset} tem esse risco?")
                                fig_waterfall = explainer.plot_waterfall(shap_values, X_sample, index=0)
                                st.pyplot(fig_waterfall)
                                
                                # Tabela de importância
                                st.subheader("Top Fatores Contribuintes")
                                df_imp = explainer.get_feature_importance(shap_values, X_sample)
                                st.dataframe(df_imp.head(10), use_container_width=True)
                                
                                # Guardar top features para Otimização PM
                                st.session_state['top_features'] = dict(zip(df_imp['feature'].head(5), df_imp['importance'].head(5)))
                                st.session_state['selected_asset'] = selected_asset
                                
                            except Exception as e:
                                st.error(f"Erro ao calcular SHAP: {e}")

        with tab3:
            st.header("🛠️ Otimização de Manutenção (IA Generativa) [3/7]")
            
            if 'top_features' not in st.session_state:
                st.warning("⚠️ **Ação Necessária:** Vá para a aba '🔍 Explicabilidade' e analise um ativo primeiro.")
                st.info("💡 Fluxo recomendado: Predições → Explicabilidade → Otimização PM")
            else:
                from src.maintenance.pm_optimizer import PMOptimizer
                import os
                
                st.markdown("Use IA Generativa para criar um plano de manutenção personalizado.")
                
                api_key = st.text_input("Gemini API Key (opcional se configurado em .env)", type="password")
                if api_key:
                    os.environ["GEMINI_API_KEY"] = api_key
                
                if st.button("Gerar Plano de Manutenção"):
                    with st.spinner("Consultando o Oráculo de Manutenção (Gemini)..."):
                        try:
                            optimizer = PMOptimizer()
                            
                            # Dados para o prompt
                            asset_name = st.session_state['selected_asset']
                            asset_info = df_raw[df_raw['ativo_unico'] == asset_name].iloc[0].to_dict()
                            
                            # Pegar predição
                            pred_row = st.session_state['df_predictions'][
                                st.session_state['df_predictions']['ativo_unico'] == asset_name
                            ].iloc[0]
                            
                            prediction_info = {
                                'horizonte': 30,
                                'probabilidade': pred_row['ProbML_Media_30d'],
                                'risco': pred_row['Classe_30d']
                            }
                            
                            plan = optimizer.generate_plan(
                                asset_info,
                                prediction_info,
                                st.session_state['top_features']
                            )
                            
                            st.markdown("### 📋 Plano de Manutenção Recomendado")
                            st.markdown(plan)
                            
                        except Exception as e:
                            st.error(f"Erro ao gerar plano: {e}")

        with tab4:
            st.header("📈 Análise Exploratória")
            
            # Distribuição de TBF
            st.subheader("Distribuição de TBF (Time Between Failures)")
            
            if 'tbf' in df_features.columns:
                fig = px.histogram(
                    df_features[df_features['tbf'] > 0],
                    x='tbf',
                    nbins=50,
                    title="Distribuição de TBF (dias)",
                    labels={'tbf': 'TBF (dias)', 'count': 'Frequência'}
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Top ativos por falhas
            st.subheader("Top 10 Ativos por Quantidade de Falhas")
            
            top_ativos = df_raw['ativo_unico'].value_counts().head(10)
            
            fig = go.Figure(go.Bar(
                x=top_ativos.values,
                y=top_ativos.index,
                orientation='h',
                marker_color='indianred'
            ))
            
            fig.update_layout(
                xaxis_title="Quantidade de Falhas",
                yaxis_title="Ativo",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Curva de Confiabilidade
            st.subheader("Análise de Confiabilidade (Sobrevivência)")
            
            try:
                from src.models.reliability import ReliabilityAnalyzer
                
                analyzer = ReliabilityAnalyzer()
                analyzer.fit(df_features, time_col='tbf')
                
                fig_reliability = analyzer.plot_survival()
                st.plotly_chart(fig_reliability, use_container_width=True)
                
                st.info("💡 As curvas mostram a probabilidade de um ativo **não falhar** ao longo do tempo.")
                
            except Exception as e:
                st.warning(f"Análise de confiabilidade não disponível: {e}")
            
            # Performance Tracker (Monitoramento de Modelos)
            st.markdown("---")
            st.subheader("📊 Performance Tracker - Monitoramento de Modelos")
            
            try:
                from src.maintenance.performance_tracker import PerformanceTracker
                
                tracker = PerformanceTracker()
                summary = tracker.get_summary()
                
                if summary['total_predictions'] == 0:
                    st.info("""
                    **Ainda sem dados de tracking.**
                    
                    O Performance Tracker registra predições realizadas e compara com falhas reais para 
                    avaliar a precisão dos modelos em produção.
                    
                    Para começar a rastrear:
                    1. Gere predições regularmente
                    2. Registre falhas reais quando ocorrerem
                    3. O sistema calculará métricas automaticamente
                    """)
                else:
                    # Métricas de resumo
                    col_p1, col_p2, col_p3 = st.columns(3)
                    
                    col_p1.metric("Total de Predições", summary['total_predictions'])
                    col_p2.metric("Com Feedback", summary['with_feedback'])
                    col_p3.metric("Aguardando Feedback", summary['pending_feedback'])
                    
                    # Drift Detection
                    if summary['with_feedback'] >= 20:
                        st.markdown("**🔍 Detecção de Drift (Deterioração do Modelo)**")
                        
                        drift_results = tracker.detect_drift(
                            horizonte=30,
                            threshold_drop=config['maintenance']['performance'].get('alert_threshold_drop', 0.05)
                        )
                        
                        if drift_results['drift_detected']:
                            st.error(f"⚠️ **DRIFT DETECTADO:** {drift_results['message']}")
                            st.warning(f"F1-Score baseline: {drift_results['f1_baseline']:.2%} → atual: {drift_results['f1_current']:.2%}")
                            st.warning("💡 **Ação Recomendada:** Considere retreinar os modelos na aba '🧠 Treinamento'")
                        else:
                            st.success(f"✅ {drift_results['message']}")
                            st.info(f"F1-Score baseline: {drift_results['f1_baseline']:.2%} | atual: {drift_results['f1_current']:.2%}")
                    
                    # Métricas ao longo do tempo
                    if summary['with_feedback'] >= 5:
                        st.markdown("**📈 Métricas ao Longo do Tempo (Janela de 30 dias)**")
                        
                        df_metrics = tracker.get_metrics_over_time(horizonte=30, window_days=30)
                        
                        if not df_metrics.empty:
                            # Gráfico de linha
                            fig_metrics = go.Figure()
                            
                            fig_metrics.add_trace(go.Scatter(
                                x=df_metrics['data'],
                                y=df_metrics['f1'],
                                name='F1-Score',
                                mode='lines+markers',
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig_metrics.add_trace(go.Scatter(
                                x=df_metrics['data'],
                                y=df_metrics['precision'],
                                name='Precision',
                                mode='lines+markers',
                                line=dict(color='green', width=2, dash='dash')
                            ))
                            
                            fig_metrics.add_trace(go.Scatter(
                                x=df_metrics['data'],
                                y=df_metrics['recall'],
                                name='Recall',
                                mode='lines+markers',
                                line=dict(color='orange', width=2, dash='dash')
                            ))
                            
                            fig_metrics.update_layout(
                                title="Evolução das Métricas de Performance",
                                xaxis_title="Data",
                                yaxis_title="Score",
                                yaxis=dict(range=[0, 1]),
                                hovermode='x unified',
                                height=400
                            )
                            
                            st.plotly_chart(fig_metrics, use_container_width=True)
                            
                            st.caption(f"Baseado em {summary['with_feedback']} predições com feedback confirmado.")
                        else:
                            st.info("Ainda sem métricas temporais suficientes.")
                    
            except Exception as e:
                st.warning(f"Performance Tracker não disponível: {e}")

        with tab5:
            st.header("🧠 Treinamento de Modelos")
            
            st.markdown("""
            Execute o treinamento completo do sistema para criar novos modelos com os dados atuais.
            
            **Atenção:** Este processo pode levar vários minutos dependendo do volume de dados.
            """)
            
            # Status do treinamento
            if 'training_status' not in st.session_state:
                st.session_state['training_status'] = 'idle'
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶️ Iniciar Treinamento Completo", type="primary"):
                    st.session_state['training_status'] = 'running'
                    
                    # Salvar dados temporariamente
                    temp_data_path = Path("outputs/temp_training_data.xlsx")
                    temp_data_path.parent.mkdir(parents=True, exist_ok=True)
                    df_raw.to_excel(temp_data_path, index=False)
                    
                    # Executar treinamento em background
                    with st.spinner("Treinando modelos..."):
                        try:
                            result = subprocess.run(
                                [sys.executable, "cli_train.py", "--data", str(temp_data_path)],
                                capture_output=True,
                                text=True,
                                timeout=600  # 10 minutos máximo
                            )
                            
                            if result.returncode == 0:
                                st.session_state['training_status'] = 'success'
                                st.success("✅ Treinamento concluído com sucesso!")
                                st.code(result.stdout)
                            else:
                                st.session_state['training_status'] = 'failed'
                                st.error("❌ Falha no treinamento")
                                st.code(result.stderr)
                                
                        except subprocess.TimeoutExpired:
                            st.error("⏱️ Timeout: Treinamento excedeu 10 minutos")
                        except Exception as e:
                            st.error(f"Erro: {e}")
            
            with col2:
                status_icon = {
                    'idle': '⚪',
                    'running': '🟡',
                    'success': '🟢',
                    'failed': '🔴'
                }
                st.metric("Status", f"{status_icon[st.session_state['training_status']]} {st.session_state['training_status'].upper()}")

        with tab6:
            st.header("📄 Gerador de Relatórios")
            
            if 'df_predictions' not in st.session_state:
                st.info("Gere predições primeiro para criar relatórios.")
            else:
                from src.explainability.report_generator import ReportGenerator
                
                st.markdown("Crie relatórios executivos em HTML para compartilhamento.")
                
                if st.button("Gerar Relatório Executivo"):
                    with st.spinner("Criando relatório..."):
                        try:
                            generator = ReportGenerator()
                            report_path = generator.generate_html_report(
                                st.session_state['df_predictions'],
                                model_version=config['project']['version']
                            )
                            
                            st.success(f"✅ Relatório gerado: {report_path}")
                            
                            # Ler e disponibilizar para download
                            with open(report_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                                
                            st.download_button(
                                label="📥 Download Relatório HTML",
                                data=html_content,
                                file_name=report_path.name,
                                mime="text/html"
                            )
                            
                            # Preview
                            with st.expander("👁️ Visualizar Relatório"):
                                st.components.v1.html(html_content, height=600, scrolling=True)
                                
                        except Exception as e:
                            st.error(f"Erro ao gerar relatório: {e}")

        with tab7:
            st.header("⚙️ Configurações do Sistema")
            
            st.markdown("Ajuste os parâmetros do sistema sem editar arquivos manualmente.")
            
            # Editar config.yaml
            st.subheader("Parâmetros de Predição")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_horizontes = st.multiselect(
                    "Horizontes de Predição (dias)",
                    options=[1, 3, 7, 15, 30, 60],
                    default=config['models']['horizontes']
                )
            
            with col2:
                new_threshold = st.slider(
                    "Threshold para Alto Risco",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(config['inference']['risk_thresholds']['alto']),
                    step=0.05
                )
            
            if st.button("💾 Salvar Configurações"):
                # Atualizar config
                config['models']['horizontes'] = new_horizontes
                config['inference']['thresholds']['alto_risco'] = new_threshold
                
                # Salvar
                config_path = Path(__file__).parent / "config.yaml"
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                
                st.success("✅ Configurações salvas! Reinicie o app para aplicar.")
                st.cache_resource.clear()
        
        with tab8:
            st.header("🔬 Análise de Causa Raiz (RCA) com IA [8/8]")
            
            from src.maintenance.rca_analyzer import RCAAnalyzer
            
            st.markdown("""
            Análise avançada de causa raiz usando:
            - Detecção estatística de anomalias
            - Padrões históricos de falhas similares
            - **IA Generativa** para insights adicionais  
            - **Chat conversacional** com contexto da análise
            """)
            
            # Selecionar ativo
            ativos_disponiveis = df_raw['ativo_unico'].unique()
            ativo_selecionado = st.selectbox("Selecione o ativo para RCA:", ativos_disponiveis)
            
            df_ativo = df_features[df_features['ativo_unico'] == ativo_selecionado].copy()
            
            if len(df_ativo) > 0:
                datas_falhas = df_ativo['data'].sort_values(ascending=False)
                
                col_r1, col_r2 = st.columns([2, 1])
                with col_r1:
                    failure_date = st.selectbox("Data da falha:", datas_falhas,
                                               format_func=lambda x: x.strftime("%d/%m/%Y %H:%M"))
                with col_r2:
                    window_days = st.slider("Janela (dias)", 3, 30, 7)
                
                use_ai = st.checkbox("Enriquecer com IA Generativa", value=True)
                
                if st.button("🔍 Executar Análise RCA", type="primary"):
                    with st.spinner("Analisando..."):
                        analyzer = RCAAnalyzer(zscore_threshold=2.0)
                        rca_report = analyzer.analyze(df_ativo, failure_date, feature_names, window_days)
                        
                        if use_ai and 'error' not in rca_report:
                            failure_descs = df_ativo['descricao'].dropna().tolist() if 'descricao' in df_ativo else None
                            rca_report = analyzer.analyze_with_ai(rca_report, failure_descs, "gemini")
                        
                        st.session_state['rca_report'] = rca_report
                        st.session_state['rca_analyzer'] = analyzer
                    st.success("✓ Análise concluída!")
                
                # Exibir relatório
                if 'rca_report' in st.session_state:
                    report = st.session_state['rca_report']
                    
                    if 'error' not in report:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Anomalias", report.get('anomalies_detected', 0))
                        c2.metric("Similares", report.get('similar_historical_failures', 0))
                        c3.metric("Confiança", f"{report.get('confidence_score', 0):.0%}")
                        c4.metric("Janela", f"{report.get('analyzed_window_days', 0)}d")
                        
                        st.markdown("---")
                        st.subheader("🎯 Top Causas Raiz")
                        
                        for i, cause in enumerate(report.get('top_root_causes', [])[:5], 1):
                            with st.expander(f"#{i} - {cause['cause_feature']} (Score: {cause['probability_score']:.2f})"):
                                st.write(f"**Confiança:** {cause['confidence']} | **Severidade:** {cause['severity']}")
                                st.write(f"**Desvio:** {cause['deviation_pct']:.1f}% | **Recorrência:** {cause['historical_recurrence']}x")
                                st.markdown(f"_{cause['description']}_")
                        
                        st.subheader("💡 Recomendações")
                        for rec in report.get('recommendations', []):
                            st.markdown(f"**{rec['priority']}.** {rec['action']} (Urgência: {rec.get('urgency', 'MÉDIA')})")
                        
                        # IA Insights
                        if 'ai_analysis' in report and 'error' not in report['ai_analysis']:
                            st.markdown("---")
                            st.subheader("🤖 Insights da IA")
                            with st.expander("💬 Análise Detalhada", expanded=True):
                                st.markdown(report['ai_analysis'].get('insights', ''))
                        
                        # Chat
                        st.markdown("---")
                        st.subheader("💬 Chat com IA sobre esta Análise")
                        
                        if 'rca_chat_history' not in st.session_state:
                            st.session_state['rca_chat_history'] = []
                        
                        for msg in st.session_state['rca_chat_history']:
                            with st.chat_message("user"):
                                st.write(msg['question'])
                            with st.chat_message("assistant"):
                                st.write(msg['answer'])
                        
                        if user_q := st.chat_input("Pergunta sobre a análise..."):
                            with st.chat_message("user"):
                                st.write(user_q)
                            
                            with st.chat_message("assistant"):
                                with st.spinner("Pensando..."):
                                    resp = st.session_state['rca_analyzer'].chat_with_context(
                                        report, user_q, 
                                        st.session_state['rca_chat_history'], "gemini"
                                    )
                                    st.write(resp['answer'])
                                    st.session_state['rca_chat_history'].append(resp)
                                    if len(st.session_state['rca_chat_history']) > 10:
                                        st.session_state['rca_chat_history'] = st.session_state['rca_chat_history'][-10:]
                        
                        if st.button("🗑️ Limpar Chat"):
                            st.session_state['rca_chat_history'] = []
                            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Erro ao processar dados: {e}")
        
        with st.expander("Ver detalhes do erro"):
            import traceback
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    f"**{config['project']['name']}** v{config['project']['version']} | "
    "Desenvolvido com Streamlit + scikit-learn + XGBoost"
)
