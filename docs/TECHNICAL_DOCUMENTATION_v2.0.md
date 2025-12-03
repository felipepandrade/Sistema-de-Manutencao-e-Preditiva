# 📘 DOCUMENTAÇÃO TÉCNICA ATUALIZADA
## Sistema Preditivo de Manutenção v2.0 - REAL Implementation

**Versão:** 2.0.0 (Implementação Real)  
**Data:** 03 de Dezembro de 2025  
**Status:** ✅ Core Funcional (70% Completo)

---

## 🎯 Status de Implementação

### Funcionalidades Implementadas (v2.0)

✅ **Data Layer Completo** (100%)
- Carregamento Excel com validação
- Schemas Pandera
- Preprocessamento robusto

✅ **Feature Engineering** (100%)
- 33+ features automáticas
- Pipeline configurável
- Tratamento de valores faltantes

✅ **Modelos Clássicos** (100%)
- 6 algoritmos (RF, XGB, LGBM, GB, ExtraTrees, CatBoost)
- Calibração de probabilidades
- Otimização de threshold

✅ **Inferência Multi-Horizonte** (100%)
- Predições para 3, 7, 15, 30 dias
- Classificação de risco (Alto/Médio/Baixo)
- Versionamento de modelos

✅ **Interface Streamlit** (100%)
- Upload e validação de dados
- Visualizações interativas
- Export CSV

✅ **CLI de Treinamento** (100%)
- Pipeline end-to-end
- Logs detalhados
- Métricas completas

### Planejado para Versões Futuras

⏳ **v2.1 (Q1 2026)**
- SHAP Explainer
- API REST (FastAPI)
- Performance Tracker

⏳ **v2.2 (Q2 2026)**
- AutoML Híbrido completo
- Deep Learning (LSTM/GRU)
- PM Optimizer com IA

⏳ **v2.3 (Q3 2026)**
- Análise de Causa Raiz (RCA)
- Integração CMMS
- Dashboard Power BI

---

## 📁 Estrutura Real do Projeto

```
predictive-maintenance-system/
├── app.py                    ✅ Streamlit completo
├── cli_train.py              ✅ Training pipeline
├── config.yaml               ✅ Configuração central
├── requirements.txt          ✅ Dependências
├── README.md                 ✅ Guia de uso
│
├── src/
│   ├── data/                 ✅ 100% Completo
│   │   ├── loaders.py
│   │   ├── validators.py
│   │   └── preprocessors.py
│   │
│   ├── features/             ✅ 100% Completo
│   │   ├── engineering.py
│   │   └── target_builder.py
│   │
│   ├── models/               ✅ Core completo
│   │   ├── classical.py
│   │   └── trainer.py
│   │
│   ├── inference/            ✅ 100% Completo
│   │   └── predictor.py
│   │
│   ├── utils/                ✅ 100% Completo
│   │   ├── io.py
│   │   ├── logging_config.py
│   │   └── metrics.py
│   │
│   ├── explainability/       ⏳ Placeholder (v2.1)
│   └── maintenance/          ⏳ Placeholder (v2.1)
│
├── models/                   ✅ Versionamento automático
├── data/                     ✅ Estrutura criada
├── outputs/                  ✅ Logs e predições
└── tests/                    ⏳ Estrutura básica
```

---

## 🔧 Módulos Implementados - Detalhes

### 1. src/data/loaders.py

**Funções:**
- `load_falhas_excel()`: Carrega Excel de falhas
  - Mapeamento por nome de colunas
  - Criação de `ativo_unico`
  - Validação e conversão de datas
  - Limpeza de strings

- `load_pcm_excel()`: Carrega ordens de serviço
  - Suporta mapeamento por índice E por nome
  - Fallback robusto

**Features Especiais:**
- ✅ Preserva lógica legada do sistema antigo
- ✅ Suporte a múltiplos encodings
- ✅ Logging detalhado

### 2. src/features/engineering.py

**Classe:** `FeatureEngineeringPipeline`

**Features Geradas (33+):**

| Categoria | Features | Exemplo |
|-----------|----------|---------|
| Temporais | tbf, falhas_acumuladas, idade_ativo_dias | 7 features |
| Estatísticas | tbf_mean_Wev, tbf_std_Wev, etc. | 16 features (W=3,6,12) |
| Tendências | tbf_pct_change, volatilidade | 2 features |
| Sazonais | mes_sin/cos, trimestre_sin/cos | 4 features |
| Interações | ratios, distâncias | 6 features |

**Uso:**
```python
pipeline = FeatureEngineeringPipeline(
    rolling_windows=[3, 6, 12],
    include_sazonalidade=True
)
df_features = pipeline.fit_transform(df_raw)
```

### 3. src/models/classical.py

**Modelos Disponíveis:**
1. RandomForest
2. XGBoost
3. LightGBM
4. GradientBoosting
5. ExtraTrees
6. CatBoost (se disponível)

**Funcionalidades:**
- ✅ Calibração de probabilidades (sigmoid/isotonic)
- ✅ Otimização de threshold por F1-Score
- ✅ Métricas completas (F1, AUC, Precision, Recall)

### 4. src/models/trainer.py

**Classe:** `ModelTrainer`

**Pipeline:**
```python
trainer = ModelTrainer(config)
results = trainer.train_all_horizons(
    df_features, feature_names, horizontes,
    mask_train, mask_val, mask_test
)
```

**Output:**
- Modelos treinados para cada horizonte (3, 7, 15, 30 dias)
- Seleção automática de campeão por F1 em validação
- Métricas de teste para validação final

### 5. src/inference/predictor.py

**Classe:** `PredictorPipeline`

**Funcionalidades:**
- Carrega modelos versionados
- Predições multi-horizonte simultâneas
- Classificação de risco automática
- Intervalos de confiança

**Thresholds de Risco (configuráveis):**
- Alto Risco: prob ≥ 70%
- Médio Risco: 30% ≤ prob < 70%
- Baixo Risco: prob < 30%

---

## 🎨 Interface Streamlit

### Tabs Implementadas:

**1. 📊 Predições**
- Upload de arquivo Excel
- Filtros por horizonte (3/7/15/30 dias)
- Filtros por risco
- Gráfico de barras colorido
- Tabela detalhada
- Métricas resumidas

**2. 📈 Análise Exploratória**
- Distribuição de TBF (histograma)
- Top 10 ativos por falhas
- Visualizações Plotly interativas

**3. 💾 Exportar**
- Download CSV de predições
- Timestamp automático
- UTF-8 encoding

**Nota:** A tab "Explicabilidade (SHAP)" está planejada para v2.1

---

## 🚀 Como Usar

### 1. Instalação

```bash
cd "Sistema financeiro com gemini"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Treinamento

```bash
# Preparar dados em data/raw/falhas.xlsx

# Treinar modelos
python cli_train.py --data data/raw/falhas.xlsx

# Aguardar ~5-15 min (dependendo dos dados)
```

**Output:**
- Modelos salvos em `models/vYYYYMMDD_HHMMSS/`
- Symlink `models/latest/` atualizado
- Logs em `outputs/logs/training.log`

### 3. Inferência

```bash
streamlit run app.py
```

1. Acesse `http://localhost:8501`
2. Upload arquivo Excel
3. Aguarde processamento (~30s)
4. Visualize predições
5. Exporte CSV

---

## 📊 Arquivos de Configuração

### config.yaml

**Seções principais:**
- `paths`: Diretórios do projeto
- `features`: Configuração de features (lags, windows)
- `models`: Horizontes, modelos a treinar, calibração
- `inference`: Thresholds de risco, IC
- `logging`: Nível de logs
- `ui`: Cores e layout Streamlit

**Exemplo de ajuste:**
```yaml
inference:
  risk_thresholds:
    alto: 0.70    # Ajustar se necessário
    medio: 0.30
```

### requirements.txt

**Core (obrigatório):**
- pandas, numpy, scipy
- scikit-learn, xgboost, lightgbm, catboost
- streamlit, plotly
- PyYAML, joblib

**Opcional (comentado):**
- h2o, flaml (AutoML)
- tensorflow (Deep Learning)
- shap (Explicabilidade)

---

## 🔍 Limitações Conhecidas (v2.0)

1. **Explicabilidade:** SHAP não implementado (v2.1)
2. **AutoML:** Apenas modelos clássicos (suficiente para maioria dos casos)
3. **Deep Learning:** Não incluído (v2.2)
4. **API REST:** Não implementada (v2.1)
5. **PM Optimizer:** IA Generativa não integrada (v2.2)

**Nota:** Todas funcionam bem com os módulos implementados. As limitações são features "nice-to-have", não bloqueantes.

---

## 📈 Métricas Esperadas

### Acurácia por Horizonte (Validação)

| Horizonte | F1-Score Esperado | AUC-ROC Esperado |
|-----------|-------------------|------------------|
| 3 dias | 0.60 - 0.70 | 0.75 - 0.82 |
| 7 dias | 0.70 - 0.78 | 0.80 - 0.87 |
| 15 dias | 0.75 - 0.82 | 0.85 - 0.90 |
| 30 dias | 0.78 - 0.85 | 0.87 - 0.92 |

**Nota:** Métricas reais dependem da qualidade e quantidade de dados históricos

---

## 🐛 Troubleshooting

### Erro: "Modelos não encontrados"
```bash
# Solução: Execute o treinamento
python cli_train.py --data data/raw/falhas.xlsx
```

### Erro: "Colunas essenciais faltando"
**Causa:** Arquivo Excel sem colunas padrão

**Solução:** Verificar arquivo tem:
- Data e Hora de Início
- Equipamento/Componente
- Instalação
- Módulo

### Performance Lenta
**Soluções:**
- Reduzir `classical_models` em config.yaml
- Usar menos janelas em `rolling_windows`
- Processar menos ativos por vez

---

## 📝 Roadmap

### v2.1 (Q1 2026)
- [ ] SHAP Explainer completo
- [ ] API REST (FastAPI)
- [ ] Performance Tracker (loop feedback)
- [ ] CLI inference em lote

### v2.2 (Q2 2026)
- [ ] AutoML Híbrido (H2O/FLAML)
- [ ] Deep Learning (LSTM/GRU)
- [ ] PM Optimizer com Gemini
- [ ] Análise RCA automatizada

### v2.3 (Q3 2026)
- [ ] Integração CMMS
- [ ] Dashboard Power BI
- [ ] RUL prediction
- [ ] Alertas automáticos

---

## ✅ Conclusão

**Sistema v2.0 está FUNCIONAL e PRONTO para uso:**

✅ Core ML completo (70% das funcionalidades planejadas)  
✅ Interface profissional Streamlit  
✅ CLI de treinamento robusto  
✅ Versionamento de modelos  
✅ Documentação completa  

**Próximos Passos:**
1. Treinar com dados reais
2. Validar predições com equipe
3. Ajustar thresholds se necessário
4. Planejar features v2.1 baseado em feedback

---

**Versão:** 2.0.0 REAL  
**Última Atualização:** 03/12/2025  
**Status:** ✅ Produção-ready para core functionality
