# O que é uma função linear?
# Uma função linear é uma expressão matemática onde a saída (y) é obtida por uma soma ponderada das entradas (x),
# cada uma multiplicada por um coeficiente (peso), mais um termo constante (intercepto).
# Exemplo geral: y = w1*x1 + w2*x2 + ... + wn*xn + b
# Isso significa que, se você dobrar uma entrada, o efeito na saída também dobra (proporcionalidade).
# Funções lineares são a base da regressão linear, pois permitem modelar relações diretas e proporcionais entre variáveis.

# Este código realiza uma regressão linear simples usando scikit-learn.
# Abaixo está uma explicação detalhada de cada etapa e da matemática envolvida.

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. Criação dos dados de entrada (X) e saída (y)
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# X é uma matriz com 4 exemplos e 2 variáveis (features) cada.

# y é calculado como o produto escalar de cada linha de X com o vetor [1, 2], somando 3.
# Ou seja, para cada linha [a, b] de X:
# y = 1*a + 2*b + 3
y = np.dot(X, np.array([1, 2])) + 3
print("Fórmula usada para calcular y: y = 1*a + 2*b + 3")
print("Coeficientes reais usados: [1, 2], Intercepto real: 3")
# Exemplo dos cálculos:
# Para [1, 1]: 1*1 + 2*1 + 3 = 6
# Para [1, 2]: 1*1 + 2*2 + 3 = 8
# Para [2, 2]: 1*2 + 2*2 + 3 = 9
# Para [2, 3]: 1*2 + 2*3 + 3 = 11
# Portanto, y = [6, 8, 9, 11]

print("Dados de entrada (X):", X.flatten())
print("Saída (y):", y)
print("Dados esperados de y: [6 8 9 11]")

# 2. Treinamento do modelo de regressão linear
model = LinearRegression().fit(X, y)
# O modelo aprende os coeficientes (pesos) que melhor relacionam X com y.

# 3. Impressão dos coeficientes aprendidos
print("Coeficientes:", model.coef_)
# model.coef_ mostra os pesos aprendidos para cada variável de entrada (x1, x2). Eles indicam quanto cada variável contribui para o valor final de y.

# 4. Impressão do intercepto aprendido
print("Intercepto:", round(float(model.intercept_), 15))
# model.intercept_ é o termo constante (bias) aprendido pelo modelo. Ele representa o valor de y quando todas as variáveis de entrada são zero.

# 5. Avaliação do modelo usando o score R²
print("Score R²:", model.score(X, y))
# model.score(X, y) retorna o 'coeficiente de determinação' (correlação) R², que mede o quanto o modelo explica da variação dos dados. R² = 1 significa ajuste perfeito; valores próximos de 0 indicam que o modelo não explica bem os dados.

# Gerando um novo conjunto de teste
X_teste = np.array([[3, 5], [0, 0], [1, 0], [2, 1]])
# Calculando os valores reais de y para o novo conjunto, usando a mesma fórmula original
y_teste = np.dot(X_teste, np.array([1, 2])) + 3

# Fazendo previsões no novo conjunto
previsoes_teste = model.predict(X_teste)
print("Fazendo previsões no conjunto de teste:", X_teste.flatten())
print("Previsões no conjunto de teste:", previsoes_teste)
# previsoes_teste mostra o que o modelo aprendeu e como ele generaliza para dados que não viu no treino.

print("Valores reais do conjunto de teste:", y_teste)
# y_teste são os valores corretos, calculados pela fórmula original.

# Score R² no conjunto de teste
print("Score R² no teste:", model.score(X_teste, y_teste))
# O score R² no teste mostra o quanto o modelo consegue explicar a variação dos dados novos.

# Mean squared error (MSE) no conjunto de teste
mse_teste = mean_squared_error(y_teste, previsoes_teste)
print("Mean Squared Error (MSE) no teste:", mse_teste)

# Mean absolute error (MAE) no conjunto de teste
mae_teste = mean_absolute_error(y_teste, previsoes_teste)
print("Mean Absolute Error (MAE) no teste:", mae_teste)

# Matemática envolvida (explicação simples):
# A regressão linear aprende uma equação do tipo:
# y = w1*x1 + w2*x2 + b
# - w1, w2: coeficientes (pesos) aprendidos
# - b: intercepto (bias)
# - x1, x2: variáveis de entrada
# O modelo ajusta w1, w2 e b para minimizar a diferença entre os valores previstos e os valores reais de y.

# Resumo:
# O código cria dados que seguem uma relação linear conhecida, treina um modelo para aprender essa relação
# e imprime os coeficientes encontrados, mostrando que a regressão linear consegue recuperar os pesos originais.


# Adicionando um teste com dados arbitrários, o modelo conseguirá encontrar uma relação linear perfeita.
X_teste_nl = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y_teste_nl = np.array([10, 20, 30, 40])
model2 = LinearRegression().fit(X_teste_nl, y_teste_nl)
previsoes_teste_nl = model2.predict(X_teste_nl)
print("\nFazendo previsões no conjunto de teste arbitrário:", X_teste_nl.flatten())
print("Previsões no conjunto de teste arbitrário:", previsoes_teste_nl)
print("Valores reais do conjunto de teste arbitrário:", y_teste_nl)
print("Coeficientes no teste arbitrário:", model2.coef_)
print("Intercepto no teste arbitrário:", round(float(model2.intercept_), 15))
print("Score R² no teste arbitrário:", model2.score(X_teste_nl, y_teste_nl))


# Adicionando um teste com dados que não seguem uma relação linear (quadrática)
# Aqui, y = x1**2 + x2**2 + 1 (função quadrática simples)
X_teste_n2 = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y_teste_n2 = X_teste_n2[:, 0] ** 2 + X_teste_n2[:, 1] ** 2 + 1

model3 = LinearRegression().fit(X_teste_n2, y_teste_n2)
previsoes_teste_n2 = model3.predict(X_teste_n2)

print("\nTeste com função quadrática simples:")
print("X_teste_n2:", X_teste_n2.flatten())
print("Valores reais (y_teste_n2):", y_teste_n2)
print("Previsões do modelo linear:", previsoes_teste_n2)
print("Coeficientes aprendidos:", model3.coef_)
print("Intercepto aprendido:", model3.intercept_)
print("Score R² (ajuste):", model3.score(X_teste_n2, y_teste_n2))

# ----------------------------------------------------------
# EXPLICAÇÃO DOS RESULTADOS DO TESTE COM FUNÇÃO QUADRÁTICA SIMPLES

# No teste com função quadrática simples, usamos:
# X_teste_n2 = [[1, 1], [2, 2], [3, 3], [4, 4]]
# y_teste_n2 = x1**2 + x2**2 + 1
# Portanto, y_teste_n2 = [3, 9, 19, 33]

# O modelo de regressão linear tentou ajustar uma linha (ou plano) a esses dados,
# mas como a relação real é quadrática (não linear), ele não consegue capturar perfeitamente a curvatura da função.

# Resultados observados:
# - Previsões do modelo linear: [ 1. 11. 21. 31.]
# - Coeficientes aprendidos: [5. 5.]
# - Intercepto aprendido: -8.999999999999993
# - Score R² (ajuste): 0.9689922480620154

# O que isso significa?
# - O modelo encontrou coeficientes [5, 5] e intercepto -9, ou seja, ajustou a função y ≈ 5*x1 + 5*x2 - 9.
# - As previsões não batem exatamente com os valores reais, mas ficam próximas.
#   Por exemplo, para [2, 2]: real = 9, previsto = 11.
# - O score R² de ~0.97 indica que o modelo linear explica cerca de 97% da variação dos dados, mas não é perfeito.
# - Isso mostra que a regressão linear pode aproximar funções não lineares até certo ponto, mas não consegue capturar toda a complexidade de uma relação quadrática.

# Resumo:
# - O modelo linear faz o melhor ajuste possível para dados não lineares, mas não consegue representar perfeitamente funções quadráticas.
# - Para capturar relações não lineares de forma exata, é necessário usar modelos mais flexíveis, como regressão polinomial ou redes neurais.


# Apresentando as três matrizes em um gráfico 2D com pyplot
import matplotlib.pyplot as plt

plt.plot(X[:, 0], y, "ro-", label="Treinamento (real)")
plt.plot(X_teste[:, 0], y_teste, "g^-", label="Teste Linear (real)")
plt.plot(X_teste_n2[:, 0], y_teste_n2, "bs-", label="Teste Quadrático (real)")

# Adicionando as linhas previstas pelo modelo para cada conjunto
plt.plot(X[:, 0], model.predict(X), "r--", label="Treinamento (previsto)")
plt.plot(X_teste[:, 0], previsoes_teste, "g--", label="Teste Linear (previsto)")
plt.plot(
    X_teste_n2[:, 0], previsoes_teste_n2, "b--", label="Teste Quadrático (previsto)"
)

plt.xlabel("X1")
plt.ylabel("y")
plt.legend()
plt.show()
