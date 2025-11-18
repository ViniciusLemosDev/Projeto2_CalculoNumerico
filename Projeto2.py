import sys

# ============================================================
#  MÉTODO DIRETO – ELIMINAÇÃO DE GAUSS (TÓPICO 1 QUESTÃO 2)
# ============================================================

def gauss_elimination(A, b, return_steps=False):
    """
    Eliminação de Gauss com pivoteamento parcial.
    Se return_steps=True, retorna também o histórico dos passos.
    """
    n = len(b)
    A = [row[:] for row in A]  # cópia defensiva
    b = b[:]
    steps = [] if return_steps else None

    try:
        for k in range(n):
            # Pivoteamento parcial
            max_row = max(range(k, n), key=lambda i: abs(A[i][k]))
            if max_row != k:
                A[k], A[max_row] = A[max_row], A[k]
                b[k], b[max_row] = b[max_row], b[k]
                if return_steps:
                    steps.append(f"Passo {k+1}: Troca de linhas {k+1} ↔ {max_row+1} (pivoteamento)")

            if A[k][k] == 0:
                raise ValueError("Sistema singular.")

            # Eliminação
            if return_steps:
                steps.append(f"Passo {k+1}: Eliminação da coluna {k+1}")
            
            for i in range(k+1, n):
                m = A[i][k] / A[k][k]
                if return_steps:
                    steps.append(f"  → Linha {i+1} = Linha {i+1} - ({m:.4f}) × Linha {k+1}")
                for j in range(k, n):
                    A[i][j] -= m * A[k][j]
                b[i] -= m * b[k]

        # Substituição regressiva
        x = [0] * n
        if return_steps:
            steps.append("Substituição regressiva:")
        for i in range(n-1, -1, -1):
            s = sum(A[i][j] * x[j] for j in range(i+1, n))
            x[i] = (b[i] - s) / A[i][i]
            if return_steps:
                steps.append(f"  x{i+1} = ({b[i]:.4f} - {s:.4f}) / {A[i][i]:.4f} = {x[i]:.4f}")

        if return_steps:
            return x, steps
        return x

    except Exception as e:
        print("Erro na eliminação de Gauss:", e)
        return (None, steps) if return_steps else None


def modulo_topico1_questao2():
    print("\n=== TÓPICO 1 – QUESTÃO 2 ===")
    print("Associação entre materiais e componentes.")
    print("Deseja usar os valores do enunciado? (S/N)")
    op = input("> ").strip().upper()

    try:
        if op == "S":
            # valores do enunciado: convertidos para kg
            A = [
                [0.015, 0.017, 0.019],     # metal por componente
                [0.00030, 0.00040, 0.00055], # plástico por componente
                [0.0010, 0.0012, 0.0015]   # borracha por componente
            ]
            b = [3.89, 0.095, 0.282]  # kg disponíveis

        else:
            print("Digite a matriz 3x3:")
            A = []
            for i in range(3):
                linha = list(map(float, input(f"Linha {i+1}: ").split()))
                if len(linha) != 3:
                    raise ValueError("Cada linha deve ter 3 valores.")
                A.append(linha)

            b = list(map(float, input("Digite o vetor b (3 valores): ").split()))
            if len(b) != 3:
                raise ValueError("b deve ter 3 valores.")

        sol = gauss_elimination(A, b)

        if sol:
            print("\nSolução do sistema (componentes produzidos por dia):")
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
    n = len(b)
    x = x0[:]

    try:
        for it in range(max_iter):
            x_old = x.copy()

            for i in range(n):
                s1 = sum(A[i][j] * x[j] for j in range(i))
                s2 = sum(A[i][j] * x_old[j] for j in range(i+1, n))
                x[i] = (b[i] - s1 - s2) / A[i][i]

            erro = max(abs(x[i] - x_old[i]) for i in range(n))
            if erro < tol:
                return x, it+1

        return x, max_iter

    except Exception as e:
        print("Erro no método de Gauss-Seidel:", e)
        return None, None


def modulo_topico2_questao3():
    print("\n=== TÓPICO 2 – QUESTÃO 3 ===")
    print("Deseja usar o sistema montado do enunciado do circuito? (S/N)")

    op = input("> ").strip().upper()

    try:
        if op == "S":
            # Sistema já montado do circuito COM a orientação da imagem
            # (Se quiser eu monto as equações aqui também — só pedir!)
            print("\n⚠ O sistema do circuito ainda não foi montado.\n"
                  "Se você quiser, eu monto as 5 equações e coloco aqui.\n")
            input("Pressione ENTER para voltar...")
            return

        else:
            n = int(input("Quantas equações? "))

            A = []
            for i in range(n):
                linha = list(map(float, input(f"Linha {i+1}: ").split()))
                if len(linha) != n:
                    raise ValueError("Cada linha deve ter N coeficientes.")
                A.append(linha)

            b = list(map(float, input("Vetor b: ").split()))
            if len(b) != n:
                raise ValueError("b deve ter N valores.")

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
    total = 0
    try:
        n = len(x)
        for i in range(n):
            L = 1
            for j in range(n):
                if i != j:
                    L *= (x0 - x[j]) / (x[i] - x[j])
            total += L * y[i]
    except:
        return None
    return total


def newton_interp(x, y, x0):
    """
    Interpolação de Newton usando diferenças divididas.
    Retorna o valor interpolado e as diferenças divididas para visualização.
    """
    try:
        n = len(x)
        # Calcular diferenças divididas
        dd = [y[i] for i in range(n)]  # Primeira ordem
        
        # Construir tabela de diferenças divididas
        for j in range(1, n):
            for i in range(n-1, j-1, -1):
                dd[i] = (dd[i] - dd[i-1]) / (x[i] - x[i-j])
        
        # Calcular valor interpolado
        result = dd[0]
        produto = 1.0
        for i in range(1, n):
            produto *= (x0 - x[i-1])
            result += dd[i] * produto
        
        return result, dd
    except:
        return None, None


def modulo_topico3_questao2():
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
            raise ValueError("Listas devem ter o mesmo tamanho.")

        x0 = float(input("Valor a interpolar: "))

        print("\nResultados pela Interpolação de Lagrange:")
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
    try:
        h = x[1] - x[0]
        return h * (y[0] + y[-1] + 2*sum(y[1:-1])) / 2
    except:
        return None

def simpson_repetido(x, y):
    try:
        h = x[1] - x[0]
        if (len(x)-1) % 2 != 0:
            raise ValueError("Simpson requer número par de intervalos.")

        soma = y[0] + y[-1]
        soma += 4 * sum(y[i] for i in range(1, len(x)-1, 2))
        soma += 2 * sum(y[i] for i in range(2, len(x)-2, 2))
        return h * soma / 3
    except:
        return None


def modulo_topico4_questao3():
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

        print("\nÁrea pelo Trapézio:", Atrap)
        print("Área por Simpson:", Asimp)

    except Exception as e:
        print("Erro:", e)

    input("\nPressione ENTER para voltar ao menu...")


# ============================================================
#  MENU PRINCIPAL — (CORRIGIDO)
# ============================================================

def menu():
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


# **ESSA LINHA É FUNDAMENTAL PARA NÃO TRAVAR O STREAMLIT**
if __name__ == "__main__":
    menu()
