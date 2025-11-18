import sys

# ============================================================
#  MÉTODO DIRETO – ELIMINAÇÃO DE GAUSS (TÓPICO 1 QUESTÃO 2)
# ============================================================

def gauss_elimination(A, b, return_steps=False):
    """
    Função que resolve sistemas lineares usando
    a Eliminação de Gauss com pivoteamento parcial.
    Se return_steps=True, retorna também o histórico detalhado.
    """

    # Número de equações
    n = len(b)

    # Realizamos cópias para evitar alterar os dados originais
    A = [row[:] for row in A]
    b = b[:]

    # Lista de passos caso o usuário deseje ver explicação
    steps = [] if return_steps else None

    try:
        # ------------------------------
        # FASE DE ELIMINAÇÃO PROGRESSIVA
        # ------------------------------
        for k in range(n):

            # Seleção do maior pivô da coluna (pivoteamento parcial)
            max_row = max(range(k, n), key=lambda i: abs(A[i][k]))

            # Se necessário, troca a linha atual pela linha com maior pivô
            if max_row != k:
                A[k], A[max_row] = A[max_row], A[k]
                b[k], b[max_row] = b[max_row], b[k]

                if return_steps:
                    steps.append(f"Passo {k+1}: Troca das linhas {k+1} ↔ {max_row+1} para melhorar o pivô.")

            # Verificação de pivô nulo (sistema singular)
            if A[k][k] == 0:
                raise ValueError("Sistema singular – divisão por zero no pivô.")

            # Explicação do passo atual
            if return_steps:
                steps.append(f"Passo {k+1}: Eliminação dos elementos abaixo do pivô na coluna {k+1}.")

            # Eliminação das linhas abaixo do pivô
            for i in range(k+1, n):
                # Cálculo do multiplicador
                m = A[i][k] / A[k][k]

                if return_steps:
                    steps.append(f"  → L{i+1} = L{i+1} - ({m:.4f}) × L{k+1}")

                # Subtração da linha k multiplicada pelo fator m
                for j in range(k, n):
                    A[i][j] -= m * A[k][j]

                # Ajuste correspondente no vetor b
                b[i] -= m * b[k]

        # ------------------------------
        # FASE DE SUBSTITUIÇÃO REGRESSIVA
        # ------------------------------
        x = [0] * n

        if return_steps:
            steps.append("Iniciando substituição regressiva...")

        # Começa pela última equação
        for i in range(n-1, -1, -1):
            s = sum(A[i][j] * x[j] for j in range(i+1, n))
            x[i] = (b[i] - s) / A[i][i]

            if return_steps:
                steps.append(f"  x{i+1} = ({b[i]:.4f} - {s:.4f}) / {A[i][i]:.4f} = {x[i]:.4f}")

        # Retorna solução + passos, se solicitado
        if return_steps:
            return x, steps
        
        return x

    except Exception as e:
        print("Erro na eliminação de Gauss:", e)
        return (None, steps) if return_steps else None



def modulo_topico1_questao2():
    """
    Módulo interativo que resolve o sistema da Questão 2 usando
    Eliminação de Gauss. Permite escolher entre os dados do enunciado
    ou valores personalizados.
    """

    print("\n=== TÓPICO 1 – QUESTÃO 2 ===")
    print("Associação entre materiais e componentes.")
    print("Deseja usar os valores do enunciado? (S/N)")
    op = input("> ").strip().upper()

    try:
        if op == "S":
            # Matriz A representa consumo de materiais por componente
            A = [
                [0.015, 0.017, 0.019],
                [0.00030, 0.00040, 0.00055],
                [0.0010, 0.0012, 0.0015]
            ]

            # Vetor b representa materiais disponíveis
            b = [3.89, 0.095, 0.282]

        else:
            print("Digite a matriz 3x3:")
            A = []
            for i in range(3):
                linha = list(map(float, input(f"Linha {i+1}: ").split()))
                if len(linha) != 3:
                    raise ValueError("Cada linha deve conter 3 valores.")
                A.append(linha)

            b = list(map(float, input("Digite o vetor b (3 valores): ").split()))
            if len(b) != 3:
                raise ValueError("O vetor b deve conter 3 valores.")

        sol = gauss_elimination(A, b)

        if sol:
            print("\nSolução do sistema – componentes produzidos por dia:")
            print(f"Componente 1 = {sol[0]:.3f}")
            print(f"Componente 2 = {sol[1]:.3f}")
            print(f"Componente 3 = {sol[2]:.3f}")

    except ValueError as e:
        print("Entrada inválida:", e)

    input("\nPressione ENTER para voltar ao menu...")



# ============================================================
#  GAUSS-SEIDEL – TÓPICO 2 QUESTÃO 3
# ============================================================

def gauss_seidel(A, b, x0, tol=1e-4, max_iter=1000):
    """
    Implementação do método iterativo de Gauss-Seidel.
    Retorna a solução aproximada e o número de iterações.
    """

    n = len(b)
    x = x0[:]

    try:
        # Loop principal de iterações
        for it in range(max_iter):

            # Guarda o vetor anterior para medir erro
            x_old = x.copy()

            # Atualiza cada variável usando os valores mais recentes
            for i in range(n):

                # Parte da soma que usa valores já atualizados
                s1 = sum(A[i][j] * x[j] for j in range(i))

                # Parte da soma que usa valores antigos
                s2 = sum(A[i][j] * x_old[j] for j in range(i+1, n))

                # Atualização da variável
                x[i] = (b[i] - s1 - s2) / A[i][i]

            # Critério de parada: maior erro entre componentes
            erro = max(abs(x[i] - x_old[i]) for i in range(n))

            if erro < tol:
                return x, it+1

        # Caso não converja no limite de iterações
        return x, max_iter

    except Exception as e:
        print("Erro no método de Gauss-Seidel:", e)
        return None, None



def modulo_topico2_questao3():
    """
    Módulo que resolve sistemas pelo método de Gauss-Seidel.
    Permite usar o sistema do enunciado ou inserir outro.
    """

    print("\n=== TÓPICO 2 – QUESTÃO 3 ===")
    print("Deseja usar o sistema montado do circuito? (S/N)")

    op = input("> ").strip().upper()

    try:
        if op == "S":
            print("\n⚠ O sistema do circuito ainda não foi montado.\n"
                  "Se você quiser, eu monto as 5 equações para você — basta pedir.\n")
            input("Pressione ENTER para voltar...")
            return

        else:
            # Entrada do sistema linear
            n = int(input("Quantas equações? "))

            A = []
            for i in range(n):
                linha = list(map(float, input(f"Linha {i+1}: ").split()))
                if len(linha) != n:
                    raise ValueError("Cada linha deve conter N coeficientes.")
                A.append(linha)

            b = list(map(float, input("Vetor b: ").split()))
            if len(b) != n:
                raise ValueError("b deve conter N valores.")

        # Chute inicial sugerido
        x0 = [b[i] / A[i][i] for i in range(n)]

        sol, it = gauss_seidel(A, b, x0)

        if sol:
            print("\nSolução aproximada do sistema:")
            for i, val in enumerate(sol):
                print(f"x{i+1} = {val:.6f}")
            print("Iterações:", it)

    except ValueError as e:
        print("Entrada inválida:", e)

    input("\nPressione ENTER para voltar ao menu...")



# ============================================================
#  INTERPOLAÇÃO POLINOMIAL – TÓPICO 3 QUESTÃO 2
# ============================================================

def lagrange_interp(x, y, x0):
    """
    Calcula P(x0) usando o polinômio interpolador de Lagrange.
    """

    total = 0
    try:
        n = len(x)

        # Para cada polinômio base L_i(x)
        for i in range(n):
            L = 1
            for j in range(n):
                if i != j:
                    # Fórmula de L_i(x0)
                    L *= (x0 - x[j]) / (x[i] - x[j])

            total += L * y[i]

    except:
        return None

    return total



def newton_interp(x, y, x0):
    """
    Calcula o valor interpolado via método de Newton
    usando a tabela de diferenças divididas.
    Também retorna a tabela para fins didáticos.
    """

    try:
        n = len(x)

        # Tabela de diferenças divididas
        dd = [y[i] for i in range(n)]

        for j in range(1, n):
            for i in range(n-1, j-1, -1):
                dd[i] = (dd[i] - dd[i-1]) / (x[i] - x[i-j])

        # Construção de P(x0)
        result = dd[0]
        produto = 1.0

        for i in range(1, n):
            produto *= (x0 - x[i-1])
            result += dd[i] * produto

        return result, dd

    except:
        return None, None



def modulo_topico3_questao2():
    """
    Módulo de interpolação (Lagrange).
    Calcula valores aproximados para ordens 2 a 4.
    """

    print("\n=== TÓPICO 3 – QUESTÃO 2 ===")
    print("Deseja usar os pontos do enunciado? (S/N)")

    op = input("> ").strip().upper()

    try:
        if op == "S":
            x = [0.25, 0.75, 1.25, 1.5, 2.0]
            y = [-0.45, -0.60, 0.70, 1.88, 6.0]

        else:
            x = list(map(float, input("x: ").split()))
            y = list(map(float, input("y: ").split()))

        if len(x) != len(y):
            raise ValueError("As listas devem ter o mesmo tamanho.")

        x0 = float(input("Valor a interpolar: "))

        print("\nResultados da Interpolação de Lagrange:")

        # Ordena os pares pelo mais próximo de x0
        pares = list(zip(x, y))
        pares.sort(key=lambda p: abs(p[0] - x0))

        for grau in range(2, min(4, len(x)-1)+1):
            xx, yy = zip(*pares[:grau+1])
            L = lagrange_interp(xx, yy, x0)
            print(f"Ordem {grau}: {L:.6f}")

    except Exception as e:
        print("Erro:", e)

    input("\nPressione ENTER para voltar ao menu...")



# ============================================================
#  INTEGRAÇÃO NUMÉRICA – TÓPICO 4 QUESTÃO 3
# ============================================================

def trapezio_repetido(x, y):
    """
    Integração numérica pelo método dos Trapézios (composto).
    """

    try:
        h = x[1] - x[0]
        return h * (y[0] + y[-1] + 2*sum(y[1:-1])) / 2
    except:
        return None



def simpson_repetido(x, y):
    """
    Integração numérica pelo método de Simpson 1/3 (composto).
    """

    try:
        h = x[1] - x[0]

        # O método exige número par de intervalos
        if (len(x)-1) % 2 != 0:
            raise ValueError("Simpson requer número PAR de intervalos.")

        soma = y[0] + y[-1]
        soma += 4 * sum(y[i] for i in range(1, len(x)-1, 2))
        soma += 2 * sum(y[i] for i in range(2, len(x)-2, 2))

        return h * soma / 3

    except:
        return None



def modulo_topico4_questao3():
    """
    Módulo de integração numérica — área transversal do navio.
    """

    print("\n=== TÓPICO 4 – QUESTÃO 3 ===")
    print("Deseja usar os dados do enunciado do navio? (S/N)")

    op = input("> ").strip().upper()

    try:
        if op == "S":
            x = [0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
            y = [3.00, 2.92, 2.75, 2.52, 2.30, 1.84, 0.92]

        else:
            x = list(map(float, input("x: ").split()))
            y = list(map(float, input("y: ").split()))

        Atrap = trapezio_repetido(x, y)
        Asimp = simpson_repetido(x, y)

        print("\nÁrea usando Trapézio:", Atrap)
        print("Área usando Simpson:", Asimp)

    except Exception as e:
        print("Erro:", e)

    input("\nPressione ENTER para voltar ao menu...")



# ============================================================
#  MENU PRINCIPAL
# ============================================================

def menu():
    """
    Menu principal do aplicativo numérico.
    Permite acessar todos os tópicos do projeto.
    """

    while True:
        print("\n==============================")
        print("      APLICATIVO NUMÉRICO")
        print("==============================")
        print("1 – Tópico 1 – Questão 2")
        print("2 – Tópico 2 – Questão 3")
        print("3 – Tópico 3 – Questão 2")
        print("4 – Tópico 4 – Questão 3")
        print("0 – Sair")
        print("==============================")

        op = input("Escolha uma opção: ").strip()

        if op == "1":
            modulo_topico1_questao2()
        elif op == "2":
            modulo_topico2_questao3()
        elif op == "3":
            modulo_topico3_questao2()
        elif op == "4":
            modulo_topico4_questao3()
        elif op == "0":
            print("Saindo...")
            sys.exit()
        else:
            print("Opção inválida. Tente novamente.")



# Execução principal (necessário para Streamlit e execução direta)
if __name__ == "__main__":
    menu()
