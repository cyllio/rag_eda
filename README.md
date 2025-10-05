# RAG_EDA - EDA com Streamlit + OpenAI Chat

## Instalação (local)
- Python 3.13
- pip install -r requirements.txt

## Execução (local)
- Configure `.streamlit/secrets.toml` com `OPENAI_API_KEY="sk-..."`
- Rode: `streamlit run app.py`

## Execução (Streamlit Cloud)
- Faça fork do repositório
- Em Secrets do app, adicione `OPENAI_API_KEY`
- Defina comando: `streamlit run app.py`

## Funcionalidades
- Upload de CSV genérico
- EDA automatizado (tipos, NA, outliers, correlação, estatísticas, perfil)
- Gráficos automáticos
- Chat em linguagem natural (OpenAI) com memória de sessão
- Conclusões automáticas do agente

## Estrutura
- `app.py`: app Streamlit principal
- `requirements.txt`: dependências
- `.streamlit/secrets.template.toml`: exemplo de secrets

## Observações
- O recurso de perfil usa `ydata-profiling` e pode estar indisponível no Windows sem toolchain C/C++; o app continua funcionando sem ele.