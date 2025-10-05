import io
import os
import json
import time
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# OpenAI SDK (>=1.0 style)
from openai import OpenAI

# Opcional: ydata-profiling (pandas-profiling)
try:
    from ydata_profiling import ProfileReport
    HAVE_PROFILING = True
except ImportError:
    HAVE_PROFILING = False

st.set_page_config(page_title="RAG EDA - CSV Agent", layout="wide")

# Aviso sobre ydata-profiling após set_page_config
if not HAVE_PROFILING:
    st.warning("ydata-profiling não está disponível. A aba 'Perfil' será desabilitada.")

# ------------- Utilitários ------------- #

def get_openai_client():
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            st.warning("OPENAI_API_KEY não configurada em .streamlit/secrets.toml.")
            return None
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Erro ao configurar cliente OpenAI: {e}")
        return None

def df_memory_snapshot(df: pd.DataFrame, max_rows=10):
    # snapshot textual resumido do DataFrame para fornecer contexto ao LLM
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "na_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "head": df.head(max_rows).to_dict(orient="records"),
        "describe_num": df.describe(include=[np.number]).round(4).to_dict(),
    }
    return info

def detect_column_roles(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    # Não escolher automaticamente colunas binárias; usar apenas nomes comuns, se existirem
    target_col = None
    for cand in ["Class", "target", "is_fraud", "fraud"]:
        if cand in df.columns:
            target_col = cand
            break
    return numeric_cols, categorical_cols, target_col

def compute_correlations(df, numeric_cols):
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        return corr
    return None

def lof_outliers(df, numeric_cols, n_neighbors=20, contamination=0.01):
    # retorna índice booleano de outliers se possível
    usable = [c for c in numeric_cols if df[c].notna().sum() == len(df)]
    if len(usable) < 2:
        return None
    X = df[usable].values
    # normaliza
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(df)-1), contamination=contamination)
    y_pred = lof.fit_predict(Xs)  # -1 outlier, 1 inlier
    return (y_pred == -1)

def generate_conclusions(df, numeric_cols, categorical_cols, target_col, corr):
    conclusions = []
    # Tamanho e nulos
    conclusions.append(f"O dataset possui {df.shape[0]} linhas e {df.shape[1]} colunas.")
    total_na = int(df.isna().sum().sum())
    if total_na > 0:
        conclusions.append(f"Foram encontrados {total_na} valores ausentes ao todo.")
    else:
        conclusions.append("Não foram encontrados valores ausentes.")

    # Distribuição de Amount e fraude (caso exista)
    if "Amount" in df.columns:
        amt_desc = df["Amount"].describe().round(2).to_dict()
        conclusions.append(f"A coluna Amount possui mediana {amt_desc.get('50%', 'NA')} e máximo {amt_desc.get('max', 'NA')}.")

    if target_col and target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False).to_dict()
        conclusions.append(f"A coluna alvo '{target_col}' possui distribuição: {vc}.")
        if "Amount" in df.columns and df[target_col].nunique() <= 10:
            grp = df.groupby(target_col)["Amount"].describe().round(2)
            conclusions.append(f"Resumo do Amount por {target_col}: {grp.to_dict()}.")

    # Correlação
    if corr is not None:
        # destacar correlações fortes
        corr_abs = corr.abs()
        tril = corr_abs.where(~np.tril(np.ones(corr_abs.shape, dtype=bool)))
        pairs = tril.stack().sort_values(ascending=False)
        top_pairs = pairs.head(5)
        if len(top_pairs) > 0:
            conclusions.append("Pares com maior correlação (valor absoluto): " + ", ".join([f"{a}~{b}: {v:.2f}" for (a,b), v in top_pairs.items()]))

    return conclusions

def eda_report(df: pd.DataFrame):
    numeric_cols, categorical_cols, target_col = detect_column_roles(df)
    corr = compute_correlations(df, numeric_cols)
    outliers_mask = lof_outliers(df, numeric_cols)  # pode ser None

    conclusions = generate_conclusions(df, numeric_cols, categorical_cols, target_col, corr)

    eda = {
        "shape": df.shape,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
        "na_counts": df.isna().sum().to_dict(),
        "describe_num": df.describe(include=[np.number]).round(4).to_dict(),
        "corr_exists": corr is not None,
        "conclusions": conclusions,
        "snapshot": df_memory_snapshot(df, max_rows=8),
    }
    if corr is not None:
        eda["corr_matrix"] = corr.round(3).to_dict()

    if outliers_mask is not None:
        eda["outliers_fraction"] = float(outliers_mask.mean())
        eda["outliers_indices"] = np.where(outliers_mask)[0].tolist()
    else:
        eda["outliers_fraction"] = None
        eda["outliers_indices"] = []

    return eda

def default_popular_chart(df, target_col):
    # Escolhe uma análise popular:
    # 1) Se houver Amount e target binário (como Class), plote distribuição de Amount por classe
    # 2) Caso contrário, mostre top-variance numéricas (bar chart)
    if "Amount" in df.columns and target_col in df.columns and df[target_col].nunique() <= 10:
        fig = px.histogram(df, x="Amount", color=target_col, nbins=50, barmode="overlay", opacity=0.6,
                           title=f"Distribuição de Amount por {target_col}")
        fig.update_layout(legend_title_text=target_col)
        return fig, "Distribuição de Amount por classe/target"
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            return None, "Sem colunas numéricas para gráfico padrão."
        variances = df[num_cols].var(numeric_only=True).sort_values(ascending=False).head(15)
        fig = px.bar(x=variances.index, y=variances.values, labels={"x": "Feature", "y": "Variância"},
                     title="Top 15 features por variância")
        return fig, "Top features por variância"

def render_corr_heatmap(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.info("Correlação indisponível (menos de duas colunas numéricas).")
        return
    corr = df[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Matriz de Correlação")
    st.pyplot(fig)

def build_system_prompt():
    # instruções para o LLM
    return textwrap.dedent("""
    Você é um assistente de análise de dados (EDA) inteligente. Responda em português do Brasil.
    Regras:
    - Baseie-se APENAS no contexto da sessão (metadados do DataFrame, amostras, estatísticas, correlação, conclusões).
    - Se a pergunta exigir cálculo que não está no contexto, explique o que falta e proponha como calcular.
    - Use linguagem clara, objetiva e didática.
    - Quando fizer afirmações numéricas, cite a fonte do contexto (ex: describe_num, distribuição, correlação).
    - Se o usuário pedir gráfico, indique qual gráfico seria adequado e qual coluna usar; o app exibirá na aba Gráficos.
    - Inclua conclusões úteis e possíveis próximos passos.
    - Considere zeros como valores válidos; nunca descarte linhas por conterem 0.
    - Valores ausentes já foram preenchidos (numéricos=0, categóricos="").
    """)

def run_llm_chat(client: OpenAI, user_msg: str, memory: dict):
    # Monta mensagens: system + contexto EDA + histórico resumido + pergunta
    system_prompt = build_system_prompt()

    # Cortar memória se muito grande
    memory_compact = {
        "eda": {k: v for k, v in memory.get("eda", {}).items() if k in ("shape","numeric_cols","categorical_cols","target_col","na_counts","describe_num","corr_matrix","conclusions","snapshot")},
        "last_answers": memory.get("last_answers", [])[-5:],
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "CONTEXT_JSON:\n" + json.dumps(memory_compact, ensure_ascii=False)}
    ]
    messages.append({"role": "user", "content": user_msg})

    # OpenAI Responses (gpt-4o-mini é econômico; ajuste conforme necessidade)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )
    answer = completion.choices[0].message.content
    return answer

# ------------- Estado de Sessão ------------- #
if "df" not in st.session_state:
    st.session_state.df = None
if "eda" not in st.session_state:
    st.session_state.eda = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_answers" not in st.session_state:
    st.session_state.last_answers = []

# Inicialização do cliente OpenAI (pode falhar se não configurado)
client = get_openai_client()

st.title("Agente EDA para CSV com Chat (Streamlit + OpenAI)")

# ------------- Upload de CSV ------------- #
st.sidebar.header("Dados")
uploaded = st.sidebar.file_uploader("Carregue um arquivo CSV", type=["csv"])
sample_note = st.sidebar.expander("Observações")
with sample_note:
    st.markdown("- Pode carregar qualquer CSV. O agente atualiza a análise e memória.")
    st.markdown("- A análise popular padrão: distribuição de Amount por classe (se existir), senão top variância.")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep=";")
    # Preenche NAs: numéricos com 0, categóricos com string vazia
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(0)
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna("")
    st.session_state.df = df
    st.session_state.eda = eda_report(df)
    st.success(f"Arquivo carregado: {uploaded.name} | Shape: {df.shape}")
elif st.session_state.df is None:
    st.info("Carregue um CSV para iniciar. Você pode começar com 'creditcard.csv'.")

# ------------- Abas ------------- #
tab_overview, tab_stats, tab_graphs, tab_profile, tab_chat = st.tabs(
    ["Visão Geral", "Estatísticas e Correlação", "Gráficos", "Perfil (opcional)", "Chat"]
)

with tab_overview:
    st.subheader("Resumo")
    if st.session_state.df is not None:
        df = st.session_state.df
        eda = st.session_state.eda

        c1, c2, c3 = st.columns(3)
        c1.metric("Linhas", eda["shape"][0])
        c2.metric("Colunas", eda["shape"][1])
        na_total = int(sum(eda["na_counts"].values()))
        c3.metric("Total de NA", na_total)

        st.write("Colunas numéricas:", ", ".join(eda["numeric_cols"]) if eda["numeric_cols"] else "nenhuma")
        st.write("Colunas categóricas:", ", ".join(eda["categorical_cols"]) if eda["categorical_cols"] else "nenhuma")
        if eda["target_col"]:
            st.write(f"Coluna alvo detectada: {eda['target_col']}")

        st.markdown("Conclusões iniciais do agente:")
        for i, c in enumerate(eda["conclusions"], 1):
            st.write(f"{i}. {c}")

        st.markdown("Amostra dos dados")
        st.dataframe(df.head(20))

    else:
        st.info("Após carregar o CSV, veja aqui o resumo e conclusões iniciais.")

with tab_stats:
    st.subheader("Estatísticas")
    if st.session_state.df is not None:
        df = st.session_state.df
        eda = st.session_state.eda

        st.write("Describe (numérico):")
        st.dataframe(pd.DataFrame(eda["describe_num"]))

        st.write("Valores ausentes por coluna:")
        st.dataframe(pd.DataFrame(eda["na_counts"], index=["NAs"]).T)

        st.subheader("Correlação")
        render_corr_heatmap(df)
    else:
        st.info("Carregue um CSV para visualizar estatísticas.")

with tab_graphs:
    st.subheader("Gráfico Popular")
    if st.session_state.df is not None:
        df = st.session_state.df
        eda = st.session_state.eda
        fig, caption = default_popular_chart(df, eda["target_col"])
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        st.caption(caption)

        st.divider()
        st.subheader("Gráficos rápidos")
        # seletor de eixos para histogram e scatter
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            g1, g2 = st.columns(2)
            with g1:
                x_hist = st.selectbox("Histograma - coluna", num_cols, key="hist_col")
                bins = st.slider("Bins", 10, 100, 40, key="hist_bins")
                fig_h = px.histogram(df, x=x_hist, nbins=bins, title=f"Histograma de {x_hist}")
                st.plotly_chart(fig_h, use_container_width=True)
            with g2:
                if len(num_cols) >= 2:
                    x_sc = st.selectbox("Scatter - eixo X", num_cols, key="sc_x")
                    y_sc = st.selectbox("Scatter - eixo Y", [c for c in num_cols if c != x_sc], key="sc_y")
                    color_opt = st.selectbox("Cor (opcional)", ["(nenhum)"] + df.columns.tolist(), key="sc_color")
                    color_kw = {} if color_opt == "(nenhum)" else {"color": color_opt}
                    fig_s = px.scatter(df, x=x_sc, y=y_sc, title=f"Scatter {x_sc} vs {y_sc}", **color_kw)
                    st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.info("Sem colunas numéricas para gráficos rápidos.")
    else:
        st.info("Carregue um CSV para visualizar gráficos.")

with tab_profile:
    st.subheader("Relatório de Perfil (ydata-profiling)")
    if st.session_state.df is not None:
        if HAVE_PROFILING:
            with st.spinner("Gerando perfil..."):
                profile = ProfileReport(st.session_state.df, title="EDA Profile", minimal=True)
                html = profile.to_html()
                st.components.v1.html(html, height=1000, scrolling=True)
        else:
            # Fallback: tentar sweetviz
            try:
                import sweetviz as sv
                st.info("ydata-profiling indisponível; exibindo relatório Sweetviz como alternativa.")
                report = sv.analyze(st.session_state.df)
                html_path = os.path.join(".streamlit", "sweetviz_report.html")
                os.makedirs(os.path.dirname(html_path), exist_ok=True)
                report.show_html(html_path, open_browser=False)
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=900, scrolling=True)
            except ModuleNotFoundError as e:
                st.warning("Sweetviz não instalado no ambiente. Instalando 'sweetviz' e 'setuptools' no requirements resolve no deploy.")
                st.caption(f"Detalhes: {e}")
            except Exception as e:
                st.error("Relatório automático indisponível.")
                st.caption(f"Detalhes: {e}")
                st.info("Use as outras abas para EDA.")
    else:
        st.info("Carregue um CSV para gerar o perfil.")

with tab_chat:
    st.subheader("Chat com o Agente EDA")
    if st.session_state.df is None:
        st.info("Carregue um CSV para conversar com o agente sobre os dados.")
    else:
        for role, content in st.session_state.chat_history:
            if role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").write(content)

        prompt = st.chat_input("Faça perguntas sobre o dataset (ex: 'Qual a distribuição de Amount por Class?')")
        if prompt:
            st.chat_message("user").write(prompt)
            st.session_state.chat_history.append(("user", prompt))
            if client is None:
                resp = "A API da OpenAI não está configurada. Ajuste .streamlit/secrets.toml."
            else:
                try:
                    resp = run_llm_chat(
                        client=client,
                        user_msg=prompt,
                        memory={"eda": st.session_state.eda, "last_answers": st.session_state.last_answers}
                    )
                except Exception as e:
                    resp = f"Falha ao consultar o modelo: {e}"

            st.chat_message("assistant").write(resp)
            st.session_state.chat_history.append(("assistant", resp))
            st.session_state.last_answers.append(resp)