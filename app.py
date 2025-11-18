import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Projeto2 import (
    gauss_elimination,
    gauss_seidel as gs_from_lib,
    lagrange_interp,
    newton_interp,
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
    font-weight: bold;
}
.card {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.05));
    border: 1px solid rgba(255,255,255,0.03);
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    margin: 10px 0;
}
.small-muted { color: #9aa6b2; font-size: 0.9rem }
.metric { background: rgba(255,255,255,0.02); padding:8px; border-radius:6px }
.step-box {
    background: rgba(14, 165, 164, 0.1);
    border-left: 4px solid #0ea5a4;
    padding: 12px;
    margin: 8px 0;
    border-radius: 4px;
}
.formula-box {
    background: rgba(255,255,255,0.05);
    padding: 12px;
    border-radius: 6px;
    font-family: 'Courier New', monospace;
    text-align: center;
    margin: 10px 0;
}
</style>
"""

st.set_page_config(page_title="CalculusFlow - Cálculo Numérico", layout="wide", page_icon="📐")
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
    """
    Sistema do circuito elétrico baseado na descrição da imagem.
    Sistema 5x5 com equações de Kirchhoff.
    """
    # Sistema correto baseado na descrição da imagem
    A = [
        [9.5,  -2.5,  0.0, -2.0,  0.0],
        [-2.5, 10.5,  0.0, -3.0, -8.0],
        [0.0,   0.0, 15.5,  0.0, -4.0],
        [-2.0, -3.0,  0.0,  7.0, -3.0],
        [0.0,  -8.0, -4.0, -3.0, 12.0],
    ]
    b = [-12.0, -16.0, -14.0, -10.0, -30.0]
    
    derivation = [
        "**Malha 1 (i1):** Aplicando KVL na malha 1:",
        "  - Resistências: 5Ω + 2.5Ω + 2Ω = 9.5Ω (diagonal)",
        "  - Resistência compartilhada com i2: -2.5Ω",
        "  - Resistência compartilhada com i4: -2Ω",
        "  - Fonte de tensão: +12V → -12V (lado direito)",
        "  **Equação:** 9.5i₁ - 2.5i₂ - 2i₄ = -12",
        "",
        "**Malha 2 (i2):** Aplicando KVL na malha 2:",
        "  - Resistências: 2.5Ω + 5Ω + 3Ω = 10.5Ω (diagonal)",
        "  - Resistência compartilhada com i1: -2.5Ω",
        "  - Resistência compartilhada com i4 e i5: -3Ω",
        "  - Resistência compartilhada com i5: -5Ω (mas já incluído no 3Ω)",
        "  - Fonte de tensão: +16V → -16V",
        "  **Equação:** -2.5i₁ + 10.5i₂ - 3i₄ - 8i₅ = -16",
        "",
        "**Malha 3 (i3):** Aplicando KVL na malha 3:",
        "  - Resistências: 3.5Ω + 8Ω + 4Ω = 15.5Ω (diagonal)",
        "  - Resistência compartilhada com i5: -4Ω",
        "  - Fonte de tensão: +14V → -14V",
        "  **Equação:** 15.5i₃ - 4i₅ = -14",
        "",
        "**Malha 4 (i4):** Aplicando KVL na malha 4:",
        "  - Resistências: 2Ω + 3Ω + 2Ω = 7Ω (diagonal)",
        "  - Resistência compartilhada com i1: -2Ω",
        "  - Resistência compartilhada com i2 e i5: -3Ω",
        "  - Fonte de tensão: +10V → -10V",
        "  **Equação:** -2i₁ - 3i₂ + 7i₄ - 3i₅ = -10",
        "",
        "**Malha 5 (i5):** Aplicando KVL na malha 5:",
        "  - Resistências: 5Ω + 3Ω + 4Ω = 12Ω (diagonal)",
        "  - Resistência compartilhada com i2: -5Ω - 3Ω = -8Ω",
        "  - Resistência compartilhada com i3: -4Ω",
        "  - Resistência compartilhada com i4: -3Ω",
        "  - Fonte de tensão: +30V → -30V",
        "  **Equação:** -8i₂ - 4i₃ - 3i₄ + 12i₅ = -30",
    ]
    return A, b, derivation

# ===========================
# UI
# ===========================
st.title("📐 CalculusFlow - Plataforma Educacional de Cálculo Numérico")

# Sidebar
with st.sidebar:
    st.header("📚 Navegação")
    page = st.radio("Escolha uma seção:", [
        "🏠 Início",
        "1️⃣ Questão 1 — Sistemas Lineares (Gauss)",
        "2️⃣ Questão 2 — Circuito Elétrico (Gauss-Seidel)",
        "3️⃣ Questão 3 — Interpolação Polinomial",
        "4️⃣ Questão 4 — Integração Numérica"
    ])

# ---------- Início ----------
if page == "🏠 Início":
    st.header("Bem-vindo ao CalculusFlow!")
    st.markdown("""
    <div class='card'>
    <h3>📖 Sobre esta Plataforma</h3>
    <p>Esta é uma plataforma educacional interativa para aprender e praticar métodos numéricos. 
    Cada questão inclui:</p>
    <ul>
        <li>📝 Explicação teórica do método</li>
        <li>🔍 Montagem passo a passo do problema</li>
        <li>🧮 Resolução detalhada com visualizações</li>
        <li>📊 Gráficos e tabelas explicativas</li>
        <li>💡 Interpretação dos resultados</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Questão 1", "Gauss", "Sistemas Lineares")
    with col2:
        st.metric("Questão 2", "Gauss-Seidel", "Método Iterativo")
    with col3:
        st.metric("Questão 3", "Interpolação", "Lagrange & Newton")
    with col4:
        st.metric("Questão 4", "Integração", "Trapézio & Simpson")

    st.markdown("---")
    st.info("💡 **Dica:** Navegue pelas questões usando o menu lateral. Cada seção contém explicações detalhadas passo a passo!")

# ---------- Questão 1 ----------
if page == "1️⃣ Questão 1 — Sistemas Lineares (Gauss)":
    st.header("Questão 1: Sistemas de Equações Lineares - Método de Eliminação de Gauss")
    
    # Explicação do problema
    with st.expander("📖 Entenda o Problema", expanded=True):
        st.markdown("""
        <div class='card'>
        <h4>📋 Enunciado do Problema</h4>
        <p>Um engenheiro supervisiona a produção de três tipos de componentes elétricos. 
        Três tipos de material — metal, plástico e borracha — são necessários para a produção.</p>
        
        <h4>📊 Dados do Problema</h4>
        <p><strong>Quantidade de material por componente (em gramas):</strong></p>
        <ul>
            <li><strong>Componente 1:</strong> 15g metal, 0.30g plástico, 1.0g borracha</li>
            <li><strong>Componente 2:</strong> 17g metal, 0.40g plástico, 1.2g borracha</li>
            <li><strong>Componente 3:</strong> 19g metal, 0.55g plástico, 1.5g borracha</li>
        </ul>
        
        <p><strong>Materiais disponíveis por dia (em kg):</strong></p>
        <ul>
            <li>Metal: 3.89 kg = 3890 g</li>
            <li>Plástico: 0.095 kg = 95 g</li>
            <li>Borracha: 0.282 kg = 282 g</li>
        </ul>
        
        <p><strong>Pergunta:</strong> Quantos componentes de cada tipo podem ser produzidos por dia?</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Montagem do sistema
    st.markdown("---")
    st.subheader("🔧 Passo 1: Montagem do Sistema de Equações")
    
    st.markdown("""
    <div class='step-box'>
    <h4>Como montar o sistema?</h4>
    <p>Sejam x₁, x₂, x₃ o número de componentes do tipo 1, 2 e 3 produzidos, respectivamente.</p>
    <p>Para cada material, temos uma equação que relaciona o consumo total com a disponibilidade:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div class='formula-box'>
        <strong>Equação do Metal:</strong><br>
        15x₁ + 17x₂ + 19x₃ = 3890
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='formula-box'>
        <strong>Equação do Plástico:</strong><br>
        0.30x₁ + 0.40x₂ + 0.55x₃ = 95
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='formula-box'>
        <strong>Equação da Borracha:</strong><br>
        1.0x₁ + 1.2x₂ + 1.5x₃ = 282
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
        <h4>📐 Forma Matricial</h4>
        <p>O sistema pode ser escrito como <strong>Ax = b</strong>, onde:</p>
        <ul>
            <li><strong>A</strong> é a matriz 3×3 dos coeficientes</li>
            <li><strong>x</strong> é o vetor [x₁, x₂, x₃]ᵀ</li>
            <li><strong>b</strong> é o vetor [3890, 95, 282]ᵀ</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Dados do problema
    st.markdown("---")
    st.subheader("📊 Passo 2: Dados do Problema")
    
    use_def = st.checkbox("✅ Usar valores do enunciado", value=True)
    
    if use_def:
        # Valores corretos: materiais em gramas, disponibilidade convertida para gramas
        A = [[15.0, 17.0, 19.0],
             [0.30, 0.40, 0.55],
             [1.0, 1.2, 1.5]]
        b = [3890.0, 95.0, 282.0]
        
        st.success("✅ Usando dados do enunciado (valores em gramas)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Matriz A (coeficientes):**")
            df_A = pd.DataFrame(A, 
                               columns=["Componente 1", "Componente 2", "Componente 3"],
                               index=["Metal (g)", "Plástico (g)", "Borracha (g)"])
            st.dataframe(df_A, use_container_width=True)
        
        with col2:
            st.write("**Vetor b (disponibilidade em gramas):**")
            df_b = pd.DataFrame({"Disponível (g)": b},
                               index=["Metal", "Plástico", "Borracha"])
            st.dataframe(df_b, use_container_width=True)
    else:
        st.write("Digite os valores manualmente:")
        A = []
        for i in range(3):
            row = st.text_input(f"Linha {i+1} da matriz A (3 valores separados por espaço)", 
                              key=f"t1_a{i}")
            if row:
                A.append(list(map(float, row.split())))
        b_str = st.text_input("Vetor b (3 valores separados por espaço)", key='t1_b')
        b = list(map(float, b_str.split())) if b_str else None

    # Resolução
    st.markdown("---")
    st.subheader("🧮 Passo 3: Resolução pelo Método de Eliminação de Gauss")
    
    with st.expander("📚 Teoria: Método de Eliminação de Gauss", expanded=False):
        st.markdown("""
        <div class='card'>
        <h4>O que é o Método de Gauss?</h4>
        <p>O método de eliminação de Gauss transforma o sistema Ax = b em um sistema triangular equivalente Ux = c, 
        onde U é uma matriz triangular superior. O processo envolve:</p>
        <ol>
            <li><strong>Pivoteamento parcial:</strong> Trocar linhas para colocar o maior elemento (em valor absoluto) na diagonal</li>
            <li><strong>Eliminação:</strong> Zerar elementos abaixo da diagonal usando operações elementares</li>
            <li><strong>Substituição regressiva:</strong> Resolver o sistema triangular de baixo para cima</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    show_steps = st.checkbox("🔍 Mostrar resolução passo a passo", value=False)
    
    if st.button("🚀 Resolver Sistema", type="primary"):
        try:
            if not use_def and (not A or not b):
                st.error("⚠️ Por favor, preencha todos os valores ou use os dados do enunciado.")
            else:
                if show_steps:
                    sol, steps = gauss_elimination(A, b, return_steps=True)
                    if sol is None:
                        st.error("❌ Sistema singular ou erro na resolução.")
                    else:
                        st.success("✅ Sistema resolvido com sucesso!")
                        
                        # Mostrar passos
                        st.markdown("### 📝 Passos da Eliminação de Gauss")
                        for step in steps:
                            st.text(step)
                        
                        # Mostrar solução
                        st.markdown("### ✅ Solução Final")
                        df_sol = pd.DataFrame({
                            "Componente": [1, 2, 3],
                            "Quantidade Produzida": [f"{s:.2f}" for s in sol],
                            "Unidade": ["unidades", "unidades", "unidades"]
                        })
                        st.dataframe(df_sol, use_container_width=True)
                        
                        # Verificação
                        st.markdown("### 🔍 Verificação")
                        A_np = np.array(A)
                        b_np = np.array(b)
                        sol_np = np.array(sol)
                        residual = np.dot(A_np, sol_np) - b_np
                        st.write(f"**Resíduo (Ax - b):** {residual}")
                        st.write(f"**Norma do resíduo:** {np.linalg.norm(residual):.2e}")
                else:
                    sol = gauss_elimination(A, b)
                    if sol is None:
                        st.error("❌ Sistema singular ou erro na resolução.")
                    else:
                        st.success("✅ Sistema resolvido com sucesso!")
                        
                        # Mostrar solução
                        st.markdown("### ✅ Solução Final")
                        df_sol = pd.DataFrame({
                            "Componente": [1, 2, 3],
                            "Quantidade Produzida": [f"{s:.2f}" for s in sol],
                            "Unidade": ["unidades", "unidades", "unidades"]
                        })
                        st.dataframe(df_sol, use_container_width=True)
                        
                        # Interpretação
                        st.markdown("### 💡 Interpretação dos Resultados")
                        st.info(f"""
                        Com os materiais disponíveis, podem ser produzidos:
                        - **{sol[0]:.0f} componentes do tipo 1**
                        - **{sol[1]:.0f} componentes do tipo 2**
                        - **{sol[2]:.0f} componentes do tipo 3**
                        
                        **Total:** {sol[0]:.0f} + {sol[1]:.0f} + {sol[2]:.0f} = {sum(sol):.0f} componentes por dia
                        """)
                        
                        # Verificação
                        A_np = np.array(A)
                        b_np = np.array(b)
                        sol_np = np.array(sol)
                        residual = np.dot(A_np, sol_np) - b_np
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Resíduo máximo", f"{np.max(np.abs(residual)):.2e}")
                        with col2:
                            st.metric("Norma do resíduo", f"{np.linalg.norm(residual):.2e}")
        except Exception as e:
            st.error(f"❌ Erro: {e}")

# ---------- Questão 2 ----------
if page == "2️⃣ Questão 2 — Circuito Elétrico (Gauss-Seidel)":
    st.header("Questão 2: Circuito Elétrico - Método de Gauss-Seidel")
    
    # Explicação do problema
    with st.expander("📖 Entenda o Problema", expanded=True):
        st.markdown("""
        <div class='card'>
        <h4>🔌 Problema do Circuito Elétrico</h4>
        <p>Dado um circuito elétrico com 5 malhas, precisamos encontrar as correntes em cada malha 
        usando a Lei de Kirchhoff das Tensões (KVL).</p>
        
        <h4>⚡ Lei de Kirchhoff das Tensões (KVL)</h4>
        <p>A soma das quedas de tensão em uma malha fechada é igual à soma das fontes de tensão naquela malha.</p>
        <p><strong>Para cada malha i:</strong></p>
        <div class='formula-box'>
        Σ (Resistências × Correntes) = Σ (Fontes de Tensão)
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Montagem do sistema
    st.markdown("---")
    st.subheader("🔧 Passo 1: Montagem do Sistema de Equações")
    
    A_circ, b_circ, deriv = circuit_system()
    
    st.markdown("""
    <div class='step-box'>
    <h4>Como montar o sistema?</h4>
    <p>Para cada malha, aplicamos KVL. A corrente em cada resistor compartilhado é a diferença 
    entre as correntes das malhas adjacentes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📐 Ver derivação completa das equações"):
        for line in deriv:
            st.markdown(line)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("**Matriz A (coeficientes do sistema):**")
        df_A = pd.DataFrame(A_circ, 
                           columns=[f"i{j+1}" for j in range(5)],
                           index=[f"Malha {j+1}" for j in range(5)])
        st.dataframe(df_A, use_container_width=True)
    
    with col2:
        st.write("**Vetor b (fontes de tensão):**")
        df_b = pd.DataFrame({"Tensão (V)": b_circ},
                           index=[f"Malha {j+1}" for j in range(5)])
        st.dataframe(df_b, use_container_width=True)
    
    # Teoria do método
    st.markdown("---")
    st.subheader("📚 Passo 2: Método de Gauss-Seidel")
    
    with st.expander("📖 Teoria: Método de Gauss-Seidel", expanded=False):
        st.markdown("""
        <div class='card'>
        <h4>O que é o Método de Gauss-Seidel?</h4>
        <p>É um método iterativo para resolver sistemas lineares Ax = b. A ideia é:</p>
        <ol>
            <li>Começar com uma aproximação inicial x⁽⁰⁾</li>
            <li>Em cada iteração k, atualizar cada componente xᵢ usando os valores já atualizados:</li>
        </ol>
        <div class='formula-box'>
        xᵢ⁽ᵏ⁺¹⁾ = (bᵢ - Σⱼ₌₁ⁱ⁻¹ aᵢⱼxⱼ⁽ᵏ⁺¹⁾ - Σⱼ₌ᵢ₊₁ⁿ aᵢⱼxⱼ⁽ᵏ⁾) / aᵢᵢ
        </div>
        <p><strong>Condição de convergência:</strong> O método converge se a matriz A for diagonalmente dominante 
        ou se o raio espectral da matriz de iteração for menor que 1.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Configuração
    st.markdown("---")
    st.subheader("⚙️ Passo 3: Configuração do Método")
    
    col1, col2 = st.columns(2)
    with col1:
        tol = st.number_input("Tolerância (erro máximo)", value=0.0001, format="%.6f", step=0.0001)
        max_it = st.number_input("Máximo de iterações", min_value=10, value=1000, step=10)
    
    with col2:
        use_initial = st.checkbox("Usar aproximação inicial bi/aii (recomendado)", value=True)
        if use_initial:
            x0 = [b_circ[i]/A_circ[i][i] for i in range(len(b_circ))]
            st.info(f"**Aproximação inicial:** {[f'{x:.4f}' for x in x0]}")
        else:
            st.write("Defina aproximação inicial manualmente:")
            x0 = [st.number_input(f"i{j+1}⁽⁰⁾", value=0.0, key=f"ch_i{j}") for j in range(5)]

    # Resolução
    if st.button("🚀 Resolver pelo Método de Gauss-Seidel", type="primary"):
        try:
            sol, hist, its = gauss_seidel_with_history(A_circ, b_circ, x0=x0, tol=tol, max_iter=int(max_it))
            
            if its >= max_it and hist[-1][2] >= tol:
                st.warning(f"⚠️ Método não convergiu em {max_it} iterações. Erro final: {hist[-1][2]:.6f}")
            else:
                st.success(f"✅ Convergência alcançada em {its} iterações!")
            
            # Resultados
            st.markdown("### ✅ Solução Final (Correntes)")
            df_sol = pd.DataFrame({
                "Corrente": [f"i{j+1}" for j in range(len(sol))],
                "Valor (A)": [f"{s:.6f}" for s in sol],
                "Valor (mA)": [f"{s*1000:.2f}" for s in sol]
            })
            st.dataframe(df_sol, use_container_width=True)
            
            # Histórico de iterações
            st.markdown("### 📊 Histórico de Convergência")
            hist_df = pd.DataFrame([{
                "Iteração": h[0],
                **{f"i{j+1}": f"{h[1][j]:.6f}" for j in range(len(sol))},
                "Erro": f"{h[2]:.6f}"
            } for h in hist])
            
            st.dataframe(hist_df.tail(30).set_index("Iteração"), use_container_width=True)
            
            # Gráfico de convergência
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Gráfico das correntes
            for j in range(len(sol)):
                ax1.plot([h[0] for h in hist], [h[1][j] for h in hist], 
                        label=f"i{j+1}", marker='o', markersize=3)
            ax1.set_xlabel('Iteração')
            ax1.set_ylabel('Corrente (A)')
            ax1.set_title('Convergência das Correntes')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico do erro
            ax2.semilogy([h[0] for h in hist], [h[2] for h in hist], 'r-', linewidth=2)
            ax2.axhline(y=tol, color='g', linestyle='--', label=f'Tolerância ({tol})')
            ax2.set_xlabel('Iteração')
            ax2.set_ylabel('Erro (escala log)')
            ax2.set_title('Convergência do Erro')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

            # Interpretação
            st.markdown("### 💡 Interpretação dos Resultados")
            st.info(f"""
            As correntes nas malhas do circuito são:
            - **i₁ = {sol[0]:.4f} A** ({sol[0]*1000:.2f} mA)
            - **i₂ = {sol[1]:.4f} A** ({sol[1]*1000:.2f} mA)
            - **i₃ = {sol[2]:.4f} A** ({sol[2]*1000:.2f} mA)
            - **i₄ = {sol[3]:.4f} A** ({sol[3]*1000:.2f} mA)
            - **i₅ = {sol[4]:.4f} A** ({sol[4]*1000:.2f} mA)
            
            **Iterações necessárias:** {its}
            **Erro final:** {hist[-1][2]:.6f}
            """)
            
        except Exception as e:
            st.error(f"❌ Erro ao executar: {e}")

# ---------- Questão 3 ----------
if page == "3️⃣ Questão 3 — Interpolação Polinomial":
    st.header("Questão 3: Interpolação Polinomial - Métodos de Lagrange e Newton")
    
    # Explicação do problema
    with st.expander("📖 Entenda o Problema", expanded=True):
        st.markdown("""
        <div class='card'>
        <h4>📊 Problema de Interpolação</h4>
        <p>Dados experimentais de queda de tensão V em um resistor para diferentes valores de corrente i:</p>
        
        <table style="width:100%">
        <tr><th>Corrente i (A)</th><th>Tensão V (V)</th></tr>
        <tr><td>0.25</td><td>-0.45</td></tr>
        <tr><td>0.75</td><td>-0.60</td></tr>
        <tr><td>1.25</td><td>0.70</td></tr>
        <tr><td>1.5</td><td>1.88</td></tr>
        <tr><td>2.0</td><td>6.0</td></tr>
        </table>
        
        <p><strong>Objetivo:</strong> Estimar a tensão V para i = 1.15 A usando interpolação polinomial 
        de graus 2, 3 e 4, usando as formas de Lagrange e Newton.</p>
        
        <p><strong>💡 Dica importante:</strong> Escolha os pontos base centrados e próximos ao valor 
        a ser interpolado (i = 1.15) para obter melhor precisão!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dados
    st.markdown("---")
    st.subheader("📊 Passo 1: Dados do Problema")
    
    use_def = st.checkbox("✅ Usar pontos do enunciado", value=True)
    
    if use_def:
        X = [0.25, 0.75, 1.25, 1.5, 2.0]
        Y = [-0.45, -0.60, 0.70, 1.88, 6.0]
        st.success("✅ Usando dados do enunciado")
    else:
        xs = st.text_input("Valores de x (corrente i) separados por espaço", "0.25 0.75 1.25 1.5 2.0")
        ys = st.text_input("Valores de y (tensão V) separados por espaço", "-0.45 -0.60 0.70 1.88 6.0")
        try:
            X = list(map(float, xs.split()))
            Y = list(map(float, ys.split()))
        except:
            X = Y = None
            st.error("Erro ao processar os dados")
    
    if X and Y:
        # Visualização dos dados
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, Y, s=100, c='red', zorder=5, label='Pontos conhecidos')
        ax.axvline(x=1.15, color='green', linestyle='--', label='Valor a interpolar (i=1.15)')
        ax.set_xlabel('Corrente i (A)')
        ax.set_ylabel('Tensão V (V)')
        ax.set_title('Dados Experimentais')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Tabela de dados
        df_data = pd.DataFrame({"Corrente i (A)": X, "Tensão V (V)": Y})
        st.dataframe(df_data, use_container_width=True)
    
    # Valor a interpolar
    st.markdown("---")
    st.subheader("🎯 Passo 2: Valor a Interpolar")
    x0 = st.number_input("Valor de i para estimar V", value=1.15, step=0.01)
    
    # Teoria
    st.markdown("---")
    st.subheader("📚 Passo 3: Métodos de Interpolação")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📖 Método de Lagrange"):
            st.markdown("""
            <div class='card'>
            <h4>Interpolação de Lagrange</h4>
            <p>O polinômio interpolador de Lagrange é dado por:</p>
            <div class='formula-box'>
            P(x) = Σᵢ₌₀ⁿ Lᵢ(x) · yᵢ
            </div>
            <p>onde os polinômios de Lagrange são:</p>
            <div class='formula-box'>
            Lᵢ(x) = Πⱼ₌₀,ⱼ≠ᵢⁿ (x - xⱼ)/(xᵢ - xⱼ)
            </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with st.expander("📖 Método de Newton"):
            st.markdown("""
            <div class='card'>
            <h4>Interpolação de Newton</h4>
            <p>O polinômio interpolador de Newton usa diferenças divididas:</p>
            <div class='formula-box'>
            P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...
            </div>
            <p>onde f[x₀,...,xₖ] são as diferenças divididas de ordem k.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Resolução
    st.markdown("---")
    st.subheader("🧮 Passo 4: Resolução")
    
    if st.button("🚀 Calcular Interpolação", type="primary") and X and Y:
        try:
            # Ordenar pontos por proximidade ao valor a interpolar
            pairs = list(zip(X, Y))
            pairs.sort(key=lambda p: abs(p[0] - x0))
            
            results = []
            
            for grau in [2, 3, 4]:
                if grau + 1 <= len(pairs):
                    # Selecionar pontos mais próximos
                    xx, yy = zip(*pairs[:grau+1])
                    xx = list(xx)
                    yy = list(yy)
                    
                    # Ordenar por x para melhor visualização
                    sorted_pairs = sorted(zip(xx, yy))
                    xx, yy = zip(*sorted_pairs)
                    xx = list(xx)
                    yy = list(yy)
                    
                    # Lagrange
                    val_lagrange = lagrange_interp(xx, yy, x0)
                    
                    # Newton
                    val_newton, dd = newton_interp(xx, yy, x0)
                    
                    results.append({
                        "Grau": grau,
                        "Pontos usados": f"{len(xx)} pontos",
                        "Pontos (i)": [f"{x:.2f}" for x in xx],
                        "Lagrange": f"{val_lagrange:.6f}",
                        "Newton": f"{val_newton:.6f}" if val_newton else "Erro",
                        "Diferença": f"{abs(val_lagrange - val_newton):.2e}" if val_newton else "N/A"
                    })
            
            # Tabela de resultados
            st.markdown("### ✅ Resultados da Interpolação")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
            
            # Visualização
            st.markdown("### 📊 Visualização dos Polinômios Interpoladores")
            
            # Gerar pontos para plotagem
            x_plot = np.linspace(min(X), max(X), 200)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            for idx, grau in enumerate([2, 3, 4]):
                if grau + 1 <= len(pairs):
                    xx, yy = zip(*pairs[:grau+1])
                    xx = sorted(list(xx))
                    yy = [Y[X.index(x)] for x in xx]
                    
                    # Calcular valores interpolados
                    y_plot = [lagrange_interp(xx, yy, xp) for xp in x_plot]
                    
                    ax = axes[idx]
                    ax.scatter(X, Y, s=100, c='red', zorder=5, label='Pontos conhecidos')
                    ax.scatter(xx, yy, s=150, c='blue', marker='s', zorder=6, label='Pontos usados')
                    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'Polinômio grau {grau}')
                    ax.axvline(x=x0, color='green', linestyle='--', linewidth=2, label=f'i = {x0}')
                    
                    # Valor interpolado
                    val_interp = lagrange_interp(xx, yy, x0)
                    ax.plot(x0, val_interp, 'go', markersize=10, zorder=7, label=f'V({x0}) = {val_interp:.4f}')
                    
                    ax.set_xlabel('Corrente i (A)')
                    ax.set_ylabel('Tensão V (V)')
                    ax.set_title(f'Interpolação de Grau {grau}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interpretação
            st.markdown("### 💡 Interpretação dos Resultados")
            if results:
                st.info(f"""
                Para i = {x0} A, a tensão estimada é:
                - **Grau 2:** V ≈ {results[0]['Lagrange']} V (Lagrange) / {results[0]['Newton']} V (Newton)
                - **Grau 3:** V ≈ {results[1]['Lagrange']} V (Lagrange) / {results[1]['Newton']} V (Newton)
                - **Grau 4:** V ≈ {results[2]['Lagrange']} V (Lagrange) / {results[2]['Newton']} V (Newton)
                
                **Observação:** Ambos os métodos (Lagrange e Newton) devem produzir o mesmo resultado 
                para o mesmo conjunto de pontos, pois ambos interpolam o mesmo polinômio único de grau n.
                """)
        
        except Exception as e:
            st.error(f"❌ Erro: {e}")

# ---------- Questão 4 ----------
if page == "4️⃣ Questão 4 — Integração Numérica":
    st.header("Questão 4: Integração Numérica - Regras do Trapézio e Simpson")
    
    # Explicação do problema
    with st.expander("📖 Entenda o Problema", expanded=True):
        st.markdown("""
        <div class='card'>
        <h4>🚢 Problema da Área do Navio</h4>
        <p>Precisamos calcular a área da seção mais larga de um navio usando métodos de integração numérica.</p>
        
        <h4>📏 Dados do Problema</h4>
        <p>O diagrama mostra a meia-seção do casco do navio com:</p>
        <ul>
            <li><strong>7 intervalos</strong> de profundidade</li>
            <li><strong>Espaçamento constante:</strong> h = 0.4 m</li>
            <li><strong>Meias-larguras (ordenadas):</strong> 3.00, 2.92, 2.75, 2.52, 2.30, 1.84, 0.92, 0.00 m</li>
        </ul>
        
        <p><strong>Observação:</strong> Como temos apenas a meia-seção, a área total será o dobro da área calculada.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dados
    st.markdown("---")
    st.subheader("📊 Passo 1: Dados do Problema")
    
    use_def = st.checkbox("✅ Usar dados do enunciado", value=True)
    
    # Inicializar variáveis
    X = None
    Y = None
    
    if use_def:
        # Dados corretos: 8 pontos (0 a 7), espaçamento de 0.4m
        X = [0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
        Y = [3.00, 2.92, 2.75, 2.52, 2.30, 1.84, 0.92, 0.00]
        st.success("✅ Usando dados do enunciado")
    else:
        xs = st.text_input("Valores de x (profundidade) separados por espaço", 
                          "0 0.4 0.8 1.2 1.6 2.0 2.4 2.8")
        ys = st.text_input("Valores de y (meia-largura) separados por espaço", 
                          "3.00 2.92 2.75 2.52 2.30 1.84 0.92 0.00")
        if xs and ys:
            try:
                X = list(map(float, xs.split()))
                Y = list(map(float, ys.split()))
                if len(X) != len(Y):
                    st.error("⚠️ Os vetores X e Y devem ter o mesmo tamanho!")
                    X = Y = None
            except Exception as e:
                X = Y = None
                st.error(f"❌ Erro ao processar os dados: {e}")
    
    if X and Y and len(X) == len(Y):
        # Visualização
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(X, Y, height=0.3, alpha=0.6, color='blue', label='Meia-largura')
        ax.plot(Y, X, 'ro-', linewidth=2, markersize=8, label='Perfil do casco')
        ax.set_xlabel('Meia-largura (m)')
        ax.set_ylabel('Profundidade (m)')
        ax.set_title('Perfil da Seção do Navio (meia-seção)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        st.pyplot(fig)
        
        # Tabela
        df_data = pd.DataFrame({
            "Profundidade (m)": X,
            "Meia-largura (m)": Y
        })
        st.dataframe(df_data, use_container_width=True)
    
    # Teoria
    st.markdown("---")
    st.subheader("📚 Passo 2: Métodos de Integração Numérica")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📖 Regra do Trapézio Repetida"):
            st.markdown("""
            <div class='card'>
            <h4>Regra do Trapézio</h4>
            <p>Para n intervalos com espaçamento h:</p>
            <div class='formula-box'>
            ∫f(x)dx ≈ (h/2)[f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]
            </div>
            <p><strong>Erro:</strong> O(h²)</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with st.expander("📖 Regra de Simpson Repetida"):
            st.markdown("""
            <div class='card'>
            <h4>Regra de Simpson</h4>
            <p>Para n par de intervalos com espaçamento h:</p>
            <div class='formula-box'>
            ∫f(x)dx ≈ (h/3)[f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + f(xₙ)]
            </div>
            <p><strong>Erro:</strong> O(h⁴) - mais preciso que Trapézio!</p>
            <p><strong>Requisito:</strong> Número par de intervalos</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Resolução
    st.markdown("---")
    st.subheader("🧮 Passo 3: Cálculo da Área")
    
    if st.button("🚀 Calcular Área", type="primary"):
        if not X or not Y:
            st.error("⚠️ Por favor, defina os dados do problema primeiro (use os dados do enunciado ou insira valores manualmente).")
        elif len(X) != len(Y):
            st.error("⚠️ Os vetores X e Y devem ter o mesmo tamanho!")
        else:
            try:
                # Verificar espaçamento uniforme
                h = X[1] - X[0]
                is_uniform = all(abs(X[i+1] - X[i] - h) < 1e-6 for i in range(len(X)-1))
                
                if not is_uniform:
                    st.warning("⚠️ Espaçamento não uniforme detectado. Os métodos podem não funcionar corretamente.")
                
                # Calcular área da meia-seção
                A_trap = trapezio_repetido(X, Y)
                A_simp = simpson_repetido(X, Y)
                
                if A_trap is None:
                    st.error("❌ Erro no cálculo pela regra do Trapézio")
                if A_simp is None:
                    st.error("❌ Erro no cálculo pela regra de Simpson (verifique se há número par de intervalos)")
                
                if A_trap and A_simp:
                    # Área total (dobro da meia-seção)
                    A_trap_total = 2 * A_trap
                    A_simp_total = 2 * A_simp
                    
                    st.markdown("### ✅ Resultados")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Área meia-seção (Trapézio)", f"{A_trap:.6f} m²")
                    with col2:
                        st.metric("Área total (Trapézio)", f"{A_trap_total:.6f} m²")
                    with col3:
                        st.metric("Área meia-seção (Simpson)", f"{A_simp:.6f} m²")
                    with col4:
                        st.metric("Área total (Simpson)", f"{A_simp_total:.6f} m²")
                    
                    # Detalhamento dos cálculos
                    st.markdown("### 📝 Detalhamento dos Cálculos")
                    
                    st.markdown("#### Regra do Trapézio Repetida")
                    st.code(f"""
h = {h:.2f} m
Área = (h/2) × [y₀ + 2(y₁ + y₂ + ... + yₙ₋₁) + yₙ]
     = ({h:.2f}/2) × [{Y[0]:.2f} + 2({sum(Y[1:-1]):.2f}) + {Y[-1]:.2f}]
     = {A_trap:.6f} m² (meia-seção)
     = {A_trap_total:.6f} m² (seção completa)
                    """)
                    
                    st.markdown("#### Regra de Simpson Repetida")
                    if (len(X) - 1) % 2 == 0:
                        st.code(f"""
h = {h:.2f} m
Área = (h/3) × [y₀ + 4y₁ + 2y₂ + 4y₃ + 2y₄ + 4y₅ + 2y₆ + y₇]
     = ({h:.2f}/3) × [{Y[0]:.2f} + 4({Y[1]:.2f}) + 2({Y[2]:.2f}) + 4({Y[3]:.2f}) + 2({Y[4]:.2f}) + 4({Y[5]:.2f}) + 2({Y[6]:.2f}) + {Y[7]:.2f}]
     = {A_simp:.6f} m² (meia-seção)
     = {A_simp_total:.6f} m² (seção completa)
                        """)
                    else:
                        st.warning("⚠️ Simpson requer número par de intervalos")
                    
                    # Comparação
                    st.markdown("### 📊 Comparação dos Métodos")
                    diff = abs(A_trap_total - A_simp_total)
                    st.info(f"""
                    **Diferença entre os métodos:** {diff:.6f} m²
                    
                    A regra de Simpson geralmente fornece resultados mais precisos (erro O(h⁴)) 
                    do que a regra do Trapézio (erro O(h²)), especialmente quando a função 
                    é suave e o número de intervalos é adequado.
                    
                    **Área da seção mais larga do navio:**
                    - Pelo método do Trapézio: **{A_trap_total:.4f} m²**
                    - Pelo método de Simpson: **{A_simp_total:.4f} m²**
                    """)
            except Exception as e:
                st.error(f"❌ Erro: {e}")
