# 📐 CalculusFlow - Plataforma Educacional de Cálculo Numérico

Plataforma interativa para aprender e praticar métodos numéricos com explicações passo a passo.

## 🚀 Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar a aplicação

```bash
streamlit run app.py
```

A aplicação será aberta automaticamente no navegador em `http://localhost:8501`

## 📚 Questões Implementadas

### Questão 1: Sistemas de Equações Lineares (Eliminação de Gauss)
- Resolve sistemas lineares usando o método de eliminação de Gauss
- Problema: Produção de componentes elétricos com materiais limitados
- Inclui explicação passo a passo da montagem do sistema e resolução

### Questão 2: Circuito Elétrico (Gauss-Seidel)
- Resolve sistemas lineares usando o método iterativo de Gauss-Seidel
- Problema: Análise de circuito elétrico com 5 malhas
- Inclui visualização da convergência e histórico de iterações

### Questão 3: Interpolação Polinomial
- Implementa interpolação de Lagrange e Newton
- Problema: Estimar tensão em resistor para corrente desconhecida
- Inclui visualização dos polinômios interpoladores

### Questão 4: Integração Numérica
- Implementa regras do Trapézio e Simpson repetidas
- Problema: Calcular área da seção de um navio
- Inclui comparação entre métodos e detalhamento dos cálculos

## 🎓 Características Educacionais

- ✅ Explicações teóricas detalhadas
- ✅ Montagem passo a passo dos problemas
- ✅ Visualizações gráficas interativas
- ✅ Interpretação dos resultados
- ✅ Comparação entre métodos
- ✅ Interface intuitiva e moderna

## 📦 Dependências

- `streamlit`: Interface web
- `numpy`: Cálculos numéricos
- `pandas`: Manipulação de dados
- `matplotlib`: Visualizações gráficas

## 🛠️ Estrutura do Projeto

```
Projeto2_CalculoNumerico/
├── app.py              # Interface Streamlit principal
├── Projeto2.py         # Implementações dos métodos numéricos
├── requirements.txt    # Dependências do projeto
└── README.md           # Este arquivo
```

## 💡 Dicas de Uso

1. Use o menu lateral para navegar entre as questões
2. Cada seção contém explicações detalhadas - expanda os painéis para ver mais
3. Marque as opções "Mostrar passo a passo" para ver resoluções detalhadas
4. Use os dados do enunciado ou insira seus próprios valores para prática

## 📝 Notas

- Todos os métodos foram implementados do zero para fins educacionais
- Os resultados podem ser verificados usando bibliotecas como numpy para comparação
- A interface foi projetada para ser educacional e intuitiva

