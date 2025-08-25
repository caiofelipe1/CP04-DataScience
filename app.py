import os
import numpy as np
import pandas as pd
import streamlit as st
import scipy.stats as stats
import plotly.express as px

st.set_page_config(
    page_title="Dashboard Profissional | Análise de Dados",
    page_icon="📊",
    layout="wide",
)

LINKEDIN_URL = "https://www.linkedin.com/in/caio-felipe-a7a67322a/"

# =========================
# SIDEBAR - CONTROLES DE DESEMPENHO
# =========================
st.sidebar.header("⚡ Desempenho")
# Modo leve: usa amostra e oculta seções pesadas até o usuário solicitar
MODO_LEVE = st.sidebar.toggle("Ativar Modo Leve (recomendado)", value=True, help="Usa amostra para gráficos e evita cálculos pesados automáticos.")
USAR_AMOSTRA = st.sidebar.toggle("Usar amostra nos gráficos/testes", value=True)
TAM_AMOSTRA = st.sidebar.slider("Tamanho da amostra", 500, 20000, 3000, 500, help="Quanto maior, mais pesado.")
LIMITE_TABELA = st.sidebar.slider("Linhas para exibição de tabelas", 5, 200, 20, 5)
st.sidebar.caption("Dica: se ficar pesado, diminua a amostra e as linhas de exibição.")

# =========================
# FUNÇÕES DE SUPORTE
# =========================
@st.cache_data(show_spinner=False)
def carregar_excel() -> pd.DataFrame:
    """Carrega o Excel de data/df_selecionado.xlsx. Faz parse de datas e remove colunas 'Unnamed'."""
    caminho = os.path.join("data", "df_selecionado.xlsx")
    if not os.path.exists(caminho):
        raise FileNotFoundError("Não encontrei data/df_selecionado.xlsx. Verifique o caminho/arquivo.")

    # dtype_backend ajuda com nulos em numéricas
    df = pd.read_excel(caminho, sheet_name=0, dtype_backend="numpy_nullable")

    # Parse de possíveis colunas de data
    possiveis_datas = [c for c in df.columns if "data" in c.lower()]
    for c in possiveis_datas:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=False, infer_datetime_format=True)
        except Exception:
            pass

    # Remove colunas automáticas sem nome (ex.: 'Unnamed: 22')
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]

    return df

def identificar_colunas(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa colunas numéricas e categóricas por dtype."""
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return num, cat

@st.cache_data(show_spinner=False)
def tabela_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela com coluna, dtype e % de nulos para exibição (cacheada)."""
    info = []
    n = len(df)
    for col in df.columns:
        nulls = df[col].isna().sum()
        null_pct = (nulls / n) * 100 if n else 0.0
        info.append({"coluna": col, "dtype": str(df[col].dtype), "%_nulos": round(null_pct, 2)})
    return pd.DataFrame(info)

def estatisticas_basicas(df: pd.DataFrame, colunas_numericas: list[str]) -> pd.DataFrame:
    """Resumo descritivo com count, média, mediana, desvio, variância, min, max."""
    if not colunas_numericas:
        return pd.DataFrame()

    resumo = []
    for c in colunas_numericas:
        serie = pd.to_numeric(df[c], errors="coerce")
        resumo.append({
            "coluna": c,
            "count": int(serie.count()),
            "mean": float(serie.mean()) if serie.count() else np.nan,
            "median": float(serie.median()) if serie.count() else np.nan,
            "std": float(serie.std(ddof=1)) if serie.count() > 1 else np.nan,
            "var": float(serie.var(ddof=1)) if serie.count() > 1 else np.nan,
            "min": float(serie.min()) if serie.count() else np.nan,
            "max": float(serie.max()) if serie.count() else np.nan,
        })
    return pd.DataFrame(resumo)

def ic_media(amostra: pd.Series, alpha: float = 0.05) -> tuple[float, tuple[float, float]]:
    """IC 100*(1-alpha)% para a média (t-Student)."""
    amostra = pd.to_numeric(amostra, errors="coerce").dropna()
    n = len(amostra)
    media = amostra.mean() if n else np.nan
    s = amostra.std(ddof=1) if n > 1 else np.nan
    if n <= 1 or pd.isna(s) or s == 0:
        return float(media), (np.nan, np.nan)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    erro = t_crit * s / np.sqrt(n)
    return float(media), (float(media - erro), float(media + erro))

@st.cache_data(show_spinner=False)
def amostrar_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Retorna uma amostra aleatória do DF (cacheada)."""
    if n >= len(df):
        return df
    return df.sample(n=n, random_state=seed)

def ler_md_opcional(nome_arquivo: str, placeholder: str) -> str:
    """Lê data/<nome_arquivo>.md se existir; senão retorna placeholder."""
    caminho = os.path.join("data", nome_arquivo)
    if os.path.exists(caminho):
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return placeholder

# =========================
# IMPORTAÇÃO DA BASE
# =========================
try:
    df = carregar_excel()
except Exception as e:
    st.error(f"❌ Erro ao carregar a base: {e}")
    st.stop()

colunas_num, colunas_cat = identificar_colunas(df)

# Cria uma versão "leve" do DF para gráficos/testes
df_viz = amostrar_df(df, TAM_AMOSTRA) if USAR_AMOSTRA else df

# =========================
# ABAS
# =========================
tabs = st.tabs([
    "🏠 Home",
    "🎓 Formação & Experiência",
    "🧰 Skills",
    "📈 Análise de Dados"
])

# -------------------------
# 🏠 HOME
# -------------------------
with tabs[0]:
    st.header("Apresentação Pessoal")
    # IMPORTANTE: largura fixa da imagem para evitar tremor na página
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        # Use o seu caminho relativo. Ex.: "Images/FotoPerfil.jpeg" (como no seu código)
        IMG_PATH = "Images/FotoPerfil.jpeg"
        if os.path.exists(IMG_PATH):
            st.image(IMG_PATH, width=260)  # <- largura fixa estabiliza o layout
        else:
            st.image("https://picsum.photos/300/300", caption="Placeholder", width=260)

        # Cartão clicável para o LinkedIn
        st.markdown(
            f"""
            <a href="{LINKEDIN_URL}" target="_blank" style="
                text-decoration:none; display:block; border:1px solid #e5e7eb;
                padding:12px 16px; border-radius:12px; background:#f8fafc;
                text-align:center; font-weight:600; margin-top:12px;">
                🔗 Visitar LinkedIn
            </a>
            """,
            unsafe_allow_html=True
        )

    with col2:
        texto_home = ler_md_opcional(
            "perfil_home.md",
            "**Caio Felipe de Lima Bezerra**  \n"
            "Sou Caio Felipe, estudante de Engenharia de Software (4º semestre) na FIAP, em São Paulo. "
            "Tenho interesse em tecnologia e inovação e estou desenvolvendo minhas habilidades para criar aplicações que resolvem problemas reais. " 
            "Busco aprender rápido, trabalhar em equipe e entregar resultados com qualidade.\n\n"
            "**Objetivo Profissional:** Quero aplicar minhas competências técnicas em projetos inovadores, atuando com equipes multidisciplinares. "
            "Meu foco é criar soluções tecnológicas úteis, escaláveis e de fácil manutenção. "
            "Busco aprendizado contínuo e impacto real no negócio.\n\n"
        )
        st.markdown(texto_home)

# -------------------------
# 🎓 FORMAÇÃO & EXPERIÊNCIA
# -------------------------
with tabs[1]:
    st.header("Formação & Experiência (LinkedIn)")
    colf, cole = st.columns(2)

    # -------- Formação --------
    with colf:
        st.subheader("Formação Acadêmica")
        st.markdown("""
- **FIAP — Bacharelado em Engenharia de Software**  
  **fev/2024 – dez/2027** • São Paulo/SP  
  Competências/tecnologias: *Python, JavaScript* (entre outras).

- **Colégio Maria Imaculada “Dr Piero Roversi” — Ensino Médio**  
  **2011 – 2022**
""")

    # -------- Experiência --------
    with cole:
        st.subheader("Experiência")
        st.markdown("""
- **Monitoriação 24x7 — C6 Bank (Estágio)**  
  **fev/2025 – atual** • São Paulo/SP • **Presencial**  
  Principais tecnologias: *Splunk, Splunk Cloud, Azure DevOps, Grafana, Microsoft Excel, Google BigQuery* (e outras competências).

- **Assistente Administrativo — FenixCo (Meio período)**  
  **abr/2022 – fev/2025 (2 anos e 11 meses)** • São Paulo/SP • **Presencial**  
  **Atividades/Resultados**  
  - Atuação em **Recursos Humanos** e rotinas **administrativas**;  
  - **Organização e estruturação** de planilhas relacionadas aos colaboradores;  
  - **Contato com candidatos**, incluindo **entrevistas** para contratação;  
  - **Acompanhamento** no contato com clientes;  
  - **Visitas eventuais** em postos;  
  - **Organização** geral de documentações.
""")


# -------------------------
# 🧰 SKILLS — OPÇÃO A (Markdown simples)
# -------------------------
with tabs[2]:
    st.header("Skills (Hard & Soft)")

    texto_skills = ler_md_opcional(
        "skills.md",
        """
### Hard Skills
- **Linguagens:** Python, Java, JavaScript, C++
- **Web:** HTML5, CSS
- **Dados / Analytics:** Pandas, NumPy, Plotly, Streamlit, Estatística
- **Banco de Dados:** SQL (PostgreSQL, MySQL)
- **Observabilidade / Monitoramento:** Splunk, Splunk Cloud, Dashboards
- **Ferramentas:** Git, GitHub, Microsoft Office

### Soft Skills
- Comunicação
- Trabalho em equipe
- Gestão de tempo
- Adaptabilidade
- Resolução de problemas
- Aprendizado rápido
- Proatividade
"""
    )
    st.markdown(texto_skills)


# -------------------------
# 📈 ANÁLISE DE DADOS
# -------------------------
with tabs[3]:
    st.header("Análise de Dados")

    sub = st.tabs([
        "📦 Bases & Tipos",
        "🧮 Estatísticas",
        "🧪 Teste t (Welch)",
        "📈 Gráficos"
    ])

    # ---------- 📦 Bases & Tipos ----------
    with sub[0]:
        st.subheader("Amostra do Dataset")
        st.write("**Dimensões (completas):** ", df.shape)
        st.dataframe(df.head(LIMITE_TABELA), use_container_width=True, height=240)

        st.subheader("Tipos de Variáveis e % de Nulos")
        st.dataframe(tabela_tipos(df).head(200), use_container_width=True, height=280)

        
        st.info(
                "**Perguntas de análise sugeridas**  \n"
                "- Qual o valor médio dos pedidos por categoria/status?  \n"
                "- Existe diferença significativa entre pedidos B2B e não-B2B?  \n"
                "- Há correlação entre quantidade (Qty) e valor do pedido?  "
            )


    # ---------- 🧮 Estatísticas ----------
    with sub[1]:
        st.subheader("Resumo Descritivo")
        if st.toggle("Calcular estatísticas descritivas", value=not MODO_LEVE, key="stats_toggle"):
            if colunas_num:
                max_cols = st.slider(
                    "Máximo de colunas numéricas a resumir",
                    1, max(1, len(colunas_num)),
                    min(5, len(colunas_num)),
                    key="max_cols_stats"
                )
                cols_sel = st.multiselect(
                    "Escolha colunas numéricas (limite acima)",
                    options=colunas_num,
                    default=colunas_num[:max_cols],
                    key="cols_sel_stats"
                )[:max_cols]

                if cols_sel:
                    resumo = estatisticas_basicas(df[cols_sel], cols_sel)
                    st.dataframe(resumo, use_container_width=True, height=300)

                    # Histogramas rápidos sobre amostra
                    if st.toggle(
                        "Mostrar histogramas (amostra)",
                        value=(not MODO_LEVE and len(cols_sel) <= 2),
                        key="histo_toggle"
                    ):
                        for gc in cols_sel[:2]:
                            fig = px.histogram(df_viz, x=gc, nbins=30, title=f"Histograma — {gc}")
                            fig.update_layout(height=320)
                            st.plotly_chart(fig, use_container_width=True, key=f"hist_stats_{gc}")
                else:
                    st.info("Selecione ao menos 1 coluna numérica.")
            else:
                st.warning("Nenhuma coluna numérica detectada.")

    # ---------- 🧪 Teste t (Welch) ----------
    with sub[2]:
        st.subheader("Intervalos de Confiança & Teste de Hipótese (t de Welch)")
        st.markdown("""
            **Justificativa do Teste t de Welch**  
            O teste t de Welch foi escolhido por ser apropriado para comparar médias de dois grupos com variâncias possivelmente diferentes e tamanhos de amostra distintos.  
            """)
        if st.toggle("Executar teste entre dois grupos", value=not MODO_LEVE, key="welch_toggle"):
            col_a, col_b = st.columns(2)
            with col_a:
                metrica = st.selectbox("Métrica numérica", options=colunas_num or [], key="welch_metric")
            with col_b:
                grupo = st.selectbox(
                    "Variável categórica (ex.: Venda_B2B, Status_Pedido, Categoria)",
                    options=colunas_cat or [],
                    key="welch_group"
                )

            if metrica and grupo:
                categorias = df_viz[grupo].dropna().astype(str).unique().tolist()
                if len(categorias) < 2:
                    st.warning("A variável categórica precisa ter ao menos 2 grupos.")
                else:
                    g1, g2 = st.columns(2)
                    with g1:
                        cat1 = st.selectbox("Grupo A", options=categorias, index=0, key="welch_cat1")
                    with g2:
                        cat2 = st.selectbox("Grupo B", options=categorias, index=1 if len(categorias) > 1 else 0, key="welch_cat2")

                    df_a = pd.to_numeric(df_viz.loc[df_viz[grupo].astype(str) == cat1, metrica], errors="coerce").dropna()
                    df_b = pd.to_numeric(df_viz.loc[df_viz[grupo].astype(str) == cat2, metrica], errors="coerce").dropna()

                    min_amostra = st.slider("Tamanho mínimo por grupo", 5, 200, 20, 5, key="welch_min")

                    if len(df_a) >= min_amostra and len(df_b) >= min_amostra:
                        media_a, ic_a = ic_media(df_a)
                        media_b, ic_b = ic_media(df_b)
                        t_stat, p_val = stats.ttest_ind(df_a, df_b, equal_var=False, nan_policy="omit")

                        colx, coly, colz = st.columns(3)
                        with colx:
                            st.metric(f"Média — {cat1}", f"{media_a:,.2f}")
                            st.caption(f"IC95%: [{ic_a[0]:,.2f}, {ic_a[1]:,.2f}]")
                        with coly:
                            st.metric(f"Média — {cat2}", f"{media_b:,.2f}")
                            st.caption(f"IC95%: [{ic_b[0]:,.2f}, {ic_b[1]:,.2f}]")
                        with colz:
                            delta = media_a - media_b
                            st.metric("Diferença (A - B)", f"{delta:,.2f}")

                        st.write(f"**Teste t (Welch)** → t = {t_stat:.3f}, p-valor = {p_val:.4f}")
                        st.info("Critério 5%: p < 0.05 → diferença estatística nas médias.")

                        if st.toggle("Mostrar gráficos comparativos (amostra)", value=not MODO_LEVE, key="welch_plots"):
                            df_plot = pd.concat([
                                pd.DataFrame({metrica: df_a, grupo: f"{cat1}"}),
                                pd.DataFrame({metrica: df_b, grupo: f"{cat2}"}),
                            ], ignore_index=True)

                            fig1 = px.histogram(
                                df_plot, x=metrica, color=grupo, barmode="overlay", nbins=30,
                                title=f"Distribuição — {metrica} por {grupo} ({cat1} vs {cat2})"
                            )
                            fig1.update_layout(height=320)
                            st.plotly_chart(
                                fig1, use_container_width=True,
                                key=f"welch_hist_{metrica}_{cat1}_{cat2}"
                            )
                            fig2 = px.box(df_plot, x=grupo, y=metrica, points="outliers",
                                          title=f"Boxplot — {metrica} por {grupo}")
                            fig2.update_layout(height=320)
                            st.plotly_chart(
                            fig2, use_container_width=True,
                            key=f"welch_box_{metrica}_{cat1}_{cat2}"
                        )                    
                    else:
                        st.warning(f"Amostra insuficiente. {cat1}: n={len(df_a)} | {cat2}: n={len(df_b)}")
            else:
                st.info("Selecione uma **métrica numérica** e um **grupo categórico** para comparar dois grupos.")


    # ---------- 📈 Gráficos ----------
    with sub[3]:
        st.subheader("Exploração Visual (amostra)")
        colg1, colg2 = st.columns(2)

        with colg1:
            if colunas_num:
                num_sel = st.selectbox("Histograma — escolha a coluna numérica", options=colunas_num, key="viz_hist_col")
                fig = px.histogram(df_viz, x=num_sel, nbins=30, title=f"Histograma — {num_sel}")
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sem colunas numéricas para histograma.")

        with colg2:
            if colunas_cat and colunas_num:
                cat_sel = st.selectbox("Boxplot — variável categórica", options=colunas_cat, key="viz_box_cat")
                num_box = st.selectbox("Boxplot — métrica numérica", options=colunas_num, key="viz_box_num")
                df_box = pd.DataFrame({
                    cat_sel: df_viz[cat_sel].astype(str),
                    num_box: pd.to_numeric(df_viz[num_box], errors="coerce")
                })
                fig = px.box(df_box, x=cat_sel, y=num_box, points="outliers",
                             title=f"Boxplot — {num_box} por {cat_sel}")
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Preciso de pelo menos 1 coluna categórica e 1 numérica.")
