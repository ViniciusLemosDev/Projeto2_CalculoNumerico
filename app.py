import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Projeto2 import (
    gauss_elimination,
    gauss_seidel as gs_from_lib,
    lagrange_interp,
    trapezio_repetido,
    simpson_repetido,
)

# ===========================
# Dark theme CSS
# ===========================
DARK_CSS = """
<style>
html, body, .main {
    background-color: #0f1724 !important;
    color: #e6eef6 !important;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.stButton>button {
    background-color: #0ea5a4;
    color: #071023;
    border-radius: 6px;
}
.card {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.05));
    border: 1px solid rgba(255,255,255,0.03);
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
}
.small-muted { color: #9aa6b2; font-size: 0.9rem }
.metric { background: rgba(255,255,255,0.02); padding:8px; border-radius:6px }
</style>
"""

st.set_page_config(page_title="CalculusFlow", layout="wide", page_icon="🖤")
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ===========================
# Helper functions
# ===========================

def gauss_seidel_with_history(A, b, x0=None, tol=1e-4, max_iter=1000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    if x0 is None:
        x = np.array([b[i] / A[i, i] for i in range(n)], dtype=float)
    else:
        x = np.array(x0, dtype=float)

    history = []
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]
        err = np.max(np.abs(x - x_old))
        history.append((k, x.copy(), err))
        if err < tol:
            return x, history, k
    return x, history, max_iter


def circuit_system():
    A = [
        [9.5,  -2.5,  0.0, -2.0,  0.0],
        [-5.0, 13.5,  0.0,  0.0, -5.0],
        [0.0,  -3.5, 15.5,  0.0,  0.0],
        [-2.0, 0.0,   0.0,  7.0, -3.0],
        [0.0,  -5.0,  0.0, -3.0, 12.0],
    ]
    b = [12.0, 16.0, 14.0, 10.0, 30.0]
    derivation = [
        "Equação 1 (i1): 9.5 i1 - 2.5 i2 - 2 i4 = 12",
        "Equação 2 (i2): -5 i1 + 13.5 i2 - 5 i5 = 16",
        "Equação 3 (i3): -3.5 i2 + 15.5 i3 = 14",
        "Equação 4 (i4): -2 i1 + 7 i4 - 3 i5 = 10",
        "Equação 5 (i5): -5 i2 - 3 i4 + 12 i5 = 30",
    ]
    return A, b, derivation

# ===========================
# UI
# ===========================
st.title("🖤 CalculusFlow")

# Sidebar
with st.sidebar:
    st.header("Navegação")
    page = st.radio("Escolha uma seção:", [
        "Início",
        "Tópico 1 — Gauss (Sistemas)",
        "Tópico 2 — Circuito (Gauss-Seidel)",
        "Tópico 3 — Interpolação",
        "Tópico 4 — Integração"
    ])


# ---------- Início ----------
if page == "Início":
    st.subheader("Resumo rápido")
    st.markdown("Este app permite: \n- resolver sistemas lineares (Gauss)\n- resolver sistemas iterativamente (Gauss-Seidel)\n- interpolar por Lagrange\n- integrar por Trapézio/Simpson\n")
    st.info("Dica: use o modo personalizado para testar outros conjuntos de dados. Na aba de circuito há uma implementação completa do sistema do enunciado.")

# ---------- Tópico 1 ----------
if page == "Tópico 1 — Gauss (Sistemas)":
    st.header("Tópico 1 — Eliminação de Gauss (Método Direto)")
    st.markdown("### Teoria resumida")
    st.write("Eliminação de Gauss aplica operações elementares para triangularizar a matriz A e depois faz substituição regressiva.")

    st.markdown("---")
    st.markdown("### Exemplo do enunciado (valores em kg)")
    st.code("A = [[0.015,0.017,0.019],[0.00030,0.00040,0.00055],[0.0010,0.0012,0.0015]]\nb = [3.89,0.095,0.282]", language='python')

    col1, col2 = st.columns([2,1])
    with col1:
        use_def = st.checkbox("Usar valores do enunciado", value=True)
        if use_def:
            A = [[0.015,0.017,0.019],[0.00030,0.00040,0.00055],[0.0010,0.0012,0.0015]]
            b = [3.89,0.095,0.282]
        else:
            st.write("Digite a matriz A (3 linhas, separadas por espaço):")
            A = []
            for i in range(3):
                row = st.text_input(f"A linha {i+1}", key=f"t1_a{i}")
                if row:
                    A.append(list(map(float, row.split())))
            b_str = st.text_input("Vetor b (3 valores)", key='t1_b')
            b = list(map(float, b_str.split())) if b_str else None

        if st.button("Calcular (Gauss)", key='t1_calc'):
            try:
                sol = gauss_elimination(A, b)
                if sol is None:
                    st.error("Sistema singular ou erro na resolução.")
                else:
                    st.success("Solução (unidades produzidas):")
                    df = pd.DataFrame({"Componente": [1,2,3], "Quantidade": sol})
                    st.table(df)
                    st.info("Interpretação: resultado indica quantos componentes podem ser fabricados com os materiais disponíveis.")
            except Exception as e:
                st.error(f"Erro: {e}")

    with col2:
        st.markdown("### Passo-a-passo (opcional)")
        if st.button("Mostrar passos — Gauss (Tópico 1)"):
            try:
                M = np.array(A, float)
                bb = np.array(b, float)
                st.write("Matriz A:")
                st.write(pd.DataFrame(M))
                st.write("Vetor b:")
                st.write(bb)
                st.write("Solução por numpy (para conferência):")
                st.write(np.linalg.solve(M, bb))
            except Exception as e:
                st.error(f"Não foi possível detalhar: {e}")

# ---------- Tópico 2 ----------
if page == "Tópico 2 — Circuito (Gauss-Seidel)":
    st.header("Tópico 2 — Circuito: sistema montado + Gauss–Seidel (educativo)")
    st.markdown("### Montagem automática do sistema através das equações de Kirchhoff — enunciado")
    A_circ, b_circ, deriv = circuit_system()
    st.write("Matriz A (coeficientes):")
    st.write(pd.DataFrame(A_circ, columns=[f"i{j+1}" for j in range(5)], index=[f"eq{j+1}" for j in range(5)]))
    st.write("Vetor b:")
    st.write(pd.Series(b_circ, index=[f"eq{j+1}" for j in range(5)]))

    st.markdown("<div class='card'><strong class='small-muted'>Derivação resumida das equações (clique para expandir)</strong></div>", unsafe_allow_html=True)
    with st.expander("Mostrar derivação completa"):
        for line in deriv:
            st.write(line)

    st.markdown("---")
    st.subheader("Configuração do método")
    tol = st.number_input("Tolerância (erro máximo)", value=1e-4, format="%.1e")
    max_it = st.number_input("Máx. iterações", min_value=10, value=1000, step=10)

    if st.checkbox("Usar aproximação inicial bi/aii (recomendado)", value=True):
        x0 = [b_circ[i]/A_circ[i][i] for i in range(len(b_circ))]
    else:
        x0 = [st.number_input(f"Chute inicial i{j+1}", value=0.0, key=f"ch_i{j}") for j in range(5)]

    if st.button("Executar Gauss–Seidel (sistema do enunciado)"):
        try:
            sol, hist, its = gauss_seidel_with_history(A_circ, b_circ, x0=x0, tol=tol, max_iter=int(max_it))
            st.success(f"Convergência em {its} iterações (erro < {tol})")
            st.write(pd.DataFrame({"Corrente": [f"i{j+1}" for j in range(len(sol))], "Valor (A)": np.round(sol,6)}))

            # Iteration plot (last values)
            hist_df = pd.DataFrame([{"it": h[0], **{f"i{j+1}": h[1][j] for j in range(len(sol))}, "err": h[2]} for h in hist])
            st.subheader("Histórico (últimas 30 iterações)")
            st.dataframe(hist_df.tail(30).set_index("it"))

            # Plot convergence
            fig, ax = plt.subplots(figsize=(8,3))
            for j in range(len(sol)):
                ax.plot(hist_df["it"], hist_df[f"i{j+1}"], label=f"i{j+1}")
            ax.set_xlabel('Iteração')
            ax.set_ylabel('Correntes (A)')
            ax.legend()
            st.pyplot(fig)

            st.markdown("**Interpretação:** valores das correntes obtidos pelo método iterativo. Verifique diagonal dominante para melhor comportamento.")
        except Exception as e:
            st.error(f"Erro ao executar: {e}")

    st.markdown("---")
    st.subheader("Testar com seu próprio sistema")
    n = st.number_input("Ordem do sistema (n x n)", min_value=2, max_value=10, value=5)
    custom_A = []
    for i in range(n):
        row = st.text_input(f"A linha {i+1} (sep espaço)", key=f"customA{i}")
        if row:
            try:
                custom_A.append(list(map(float, row.split())))
            except:
                custom_A = None
    b_str = st.text_input("b (sep espaço)", key="custom_b")
    try:
        custom_b = list(map(float, b_str.split())) if b_str else None
    except:
        custom_b = None

    if st.button("Resolver sistema customizado (Gauss–Seidel)"):
        try:
            if custom_A is None or custom_b is None:
                st.error("Preencha A e b corretamente")
            else:
                x0_custom = [custom_b[i]/custom_A[i][i] for i in range(n)]
                solc, histc, itsc = gauss_seidel_with_history(custom_A, custom_b, x0=x0_custom, tol=tol, max_iter=int(max_it))
                st.write(pd.DataFrame({"x": solc}))
        except Exception as e:
            st.error(f"Erro: {e}")

# ---------- Tópico 3 ----------
if page == "Tópico 3 — Interpolação":
    st.header("Tópico 3 — Interpolação Polinomial (Lagrange)")
    st.write("Insira pontos ou use os do enunciado. O app calcula Lagrange para várias ordens e mostra diferenças.")
    use_def = st.checkbox("Usar pontos do enunciado", value=True)
    if use_def:
        X = [0.25,0.75,1.25,1.5,2.0]
        Y = [-0.45,-0.60,0.70,1.88,6.00]
    else:
        xs = st.text_input("x (sep espaço)", "0.25 0.75 1.25 1.5 2.0")
        ys = st.text_input("y (sep espaço)", "-0.45 -0.60 0.70 1.88 6.00")
        try:
            X = list(map(float, xs.split()))
            Y = list(map(float, ys.split()))
        except:
            X = Y = None

    x0 = st.number_input("Valor para estimar (x0)", value=1.15)
    if st.button("Calcular interpolação"):
        try:
            pairs = list(zip(X,Y))
            pairs.sort(key=lambda p: abs(p[0]-x0))
            out = []
            for grau in [2,3,4]:
                if grau+1 <= len(pairs):
                    xx, yy = zip(*pairs[:grau+1])
                    val = lagrange_interp(xx, yy, x0)
                    out.append((grau, val))
            st.table(pd.DataFrame(out, columns=["Grau","Valor_estimado"]))
            st.info("Escolha pontos centrados e próximos a x0 para melhor acurácia.")
        except Exception as e:
            st.error(f"Erro: {e}")

# ---------- Tópico 4 ----------
if page == "Tópico 4 — Integração":
    st.header("Tópico 4 — Integração Numérica (Trapézio e Simpson)")
    st.write("Use os dados do enunciado do navio ou insira seus próprios.")

    default_x = "0 0.4 0.8 1.2 1.6 2.0 2.4"
    default_y = "3.00 2.92 2.75 2.52 2.30 1.84 0.92"
    use_def = st.checkbox("Usar dados do enunciado", value=True)
    if use_def:
        X = list(map(float, default_x.split()))
        Y = list(map(float, default_y.split()))
    else:
        xs = st.text_input("x (sep espaço)", default_x)
        ys = st.text_input("y (sep espaço)", default_y)
        try:
            X = list(map(float, xs.split()))
            Y = list(map(float, ys.split()))
        except:
            X=Y=None

    if st.button("Calcular áreas"):
        try:
            A_trap = trapezio_repetido(X, Y)
            A_simp = simpson_repetido(X, Y)
            st.metric("Área - Trapézio (m²)", f"{A_trap:.6f}")
            st.metric("Área - Simpson (m²)", f"{A_simp:.6f}")
            st.write("Observação: Simpson requer número par de intervalos. Verifique espaçamento uniforme.")
        except Exception as e:
            st.error(f"Erro: {e}")


