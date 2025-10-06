# RAG EDA - Agente de Análise Exploratória de Dados com Chat IA

Uma aplicação Streamlit avançada que combina análise exploratória de dados (EDA) automatizada com chat inteligente usando OpenAI, perfeita para análise de arquivos CSV.

## 🚀 Funcionalidades

- **📊 Upload de CSV universal** - Suporte a diferentes encodings e separadores
- **🔍 EDA automatizado** - Análise inteligente de tipos, correlações, outliers e estatísticas
- **📈 Visualizações interativas** - Gráficos com Plotly e análises automáticas
- **🤖 Chat inteligente** - Conversa em português sobre seus dados com IA
- **📋 Relatórios detalhados** - Perfil completo com ydata-profiling e sweetviz
- **🌐 Compatibilidade total** - Funciona com arquivos brasileiros (acentos, separadores)

## 🛠️ Instalação Local

### Pré-requisitos
- Python 3.8+
- Chave da API OpenAI (opcional, para chat)

### Passos

1. **Clone o repositório:**
```bash
git clone https://github.com/cyllio/rag_eda.git
cd rag_eda
```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

3. **Configure a chave OpenAI (opcional):**
   - Crie o arquivo `.streamlit/secrets.toml`
   - Adicione: `OPENAI_API_KEY = "sua_chave_aqui"`

4. **Execute a aplicação:**
```bash
streamlit run app.py
```

## ☁️ Deploy no Streamlit Cloud

### Configuração Automática

1. **Fork este repositório** no GitHub
2. **Acesse [Streamlit Cloud](https://share.streamlit.io/)**
3. **Clique em "New app"**
4. **Configure:**
   - **Repository:** `seu-usuario/rag_eda`
   - **Branch:** `main`
   - **Main file path:** `app.py`

### Secrets (Opcional)

Para habilitar o chat com IA, adicione no Streamlit Cloud:
- **Secrets:** `OPENAI_API_KEY = "sua_chave_aqui"`

## 📋 Dependências

### Principais
- `streamlit==1.38.0` - Framework web
- `pandas==2.3.2` - Manipulação de dados
- `numpy==2.1.3` - Computação numérica
- `plotly==5.24.1` - Visualizações interativas
- `openai==2.1.0` - Chat com IA

### Análise e Visualização
- `matplotlib==3.10.0` - Gráficos estáticos
- `seaborn==0.13.2` - Visualizações estatísticas
- `scikit-learn==1.5.2` - Machine learning
- `ydata-profiling>=4.6.0` - Relatórios automáticos
- `sweetviz==2.3.1` - Análise alternativa

## 🎯 Como Usar

### 1. Upload de Dados
- Carregue qualquer arquivo CSV na sidebar
- O sistema detecta automaticamente encoding e separador
- Suporte para arquivos brasileiros com acentos

### 2. Análise Automática
- **Visão Geral:** Resumo e conclusões automáticas
- **Estatísticas:** Correlações e estatísticas descritivas
- **Gráficos:** Visualizações automáticas e personalizáveis
- **Perfil:** Relatório detalhado com ydata-profiling

### 3. Chat Inteligente
Faça perguntas em português sobre seus dados:
- "Qual a distribuição de Amount por Class?"
- "Quais são as correlações mais fortes?"
- "Quantos outliers existem?"
- "Qual o mês com maior consumo?" (para dados ANOMES)

## 🔧 Recursos Técnicos

### Detecção Automática
- **Encodings:** UTF-8, Latin-1, ISO-8859-1, CP1252
- **Separadores:** Vírgula, ponto-e-vírgula, tab
- **Tipos de dados:** Numérico, categórico, ANOMES

### Análise Inteligente
- **Outliers:** Detecção com Local Outlier Factor
- **Correlações:** Matriz de correlação automática
- **Target detection:** Identificação de colunas alvo
- **ANOMES:** Suporte para dados temporais (YYYYMM)

### Fallbacks Robustos
- ydata-profiling → sweetviz (se indisponível)
- Chat com IA → Modo sem chat (se API não configurada)
- Múltiplos encodings → Tentativas automáticas

## 📊 Exemplos de Uso

### Dados Financeiros
- Análise de transações
- Detecção de fraudes
- Correlações de mercado

### Dados Temporais (ANOMES)
- Consumo mensal
- Vendas por período
- Análises sazonais

### Dados Categóricos
- Segmentação de clientes
- Análises demográficas
- Classificações

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas alterações (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 🙏 Agradecimentos

- [Streamlit](https://streamlit.io/) - Framework web
- [ydata-profiling](https://github.com/ydataai/ydata-profiling) - Relatórios automáticos
- [OpenAI](https://openai.com/) - Chat inteligente
- [Plotly](https://plotly.com/) - Visualizações interativas

---

**Desenvolvido com ❤️ para análise de dados em português brasileiro**