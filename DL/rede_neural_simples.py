import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from keras import Input, Model
from keras.layers import Dense

# ----------------------------------------------------------
# EXPLICAÇÃO DETALHADA DO CÓDIGO
#
# Este código constrói e treina uma rede neural simples usando Keras.
#
# RESUMO:
# O código cria uma rede neural para prever um valor a partir de exemplos com 3 características (3 dimensões no vetor de entrada).
# O modelo aprende a combinar essas 3 informações para tentar prever o valor de saída, ajustando seus pesos automaticamente durante o treinamento.
# ----------------------------------------------------------

# 1. Geração dos dados:
#    - X é uma matriz de tamanho (100, 3): são 100 amostras, cada uma com 3 características (ou features).
#      Cada característica é uma informação diferente sobre o exemplo. Por exemplo, se estivéssemos prevendo o preço de uma casa,
#      as 3 características poderiam ser: metragem, número de quartos e idade do imóvel.
#      No código, cada linha de X é um vetor tridimensional (3 dimensões), representando um exemplo com 3 informações diferentes.
#    - y é um vetor de tamanho (100, 1): são 100 valores de saída, um para cada amostra.

# Exemplo de dados de entrada
# X: matriz com 200 amostras e 3 características
# y: vetor de saída com 200 valores
X = np.random.random((200, 3))


# Cálculo de y (saída) usando uma função não linear específica:
# Para cada amostra, temos 3 características: x1, x2, x3.
# A saída y é calculada assim:
#   - Eleva a primeira característica ao quadrado: x1**2
#   - Multiplica a segunda característica por 2: x2*2
#   - Calcula o seno da terceira característica: sin(x3)
#   - Soma todos esses valores: y = x1**2 + x2*2 + sin(x3)
#   - .reshape(-1, 1): garante que y tenha a forma correta (100, 1).
# Ou seja, cada linha de X gera um y diferente combinando operações não lineares e lineares.
def evaluate_polynomial(X):
    return (X[:, 0] ** 2 + 2 * X[:, 1] + np.sin(X[:, 2])).reshape(-1, 1)


y = evaluate_polynomial(X)

# 2. Definição do modelo:
#    - inputs = Input(shape=(3,)): define a camada de entrada, esperando vetores com 3 valores (as 3 características).
#    - x = Dense(15, activation="relu")(inputs): camada oculta com 15 neurônios, cada um recebendo as 3 entradas.
#      A função de ativação "relu" (Rectified Linear Unit) transforma a saída de cada neurônio para zero se o valor for negativo,
#      ou mantém o valor se for positivo. Isso ajuda a rede a aprender relações não lineares.
#    - outputs = Dense(1)(x): camada de saída com 1 neurônio, que prevê um valor contínuo para cada amostra.
#    - model = Model(inputs=inputs, outputs=outputs): conecta as camadas formando o modelo final.

# Definindo o modelo usando Input
inputs = Input(shape=(3,))
x = Dense(15, activation="relu")(inputs)
outputs = Dense(1)(x)
model = Model(inputs=inputs, outputs=outputs)

# 3. Compilação do modelo:
#    - model.compile(optimizer="adam", loss="mean_squared_error")
#      - optimizer="adam": Adam é um algoritmo de otimização muito usado em redes neurais. Ele ajusta automaticamente a taxa de aprendizado
#        para cada peso da rede, combinando as vantagens dos métodos AdaGrad e RMSProp. Isso faz com que o treinamento seja mais eficiente
#        e estável, especialmente em problemas com muitos dados ou parâmetros.
#      - loss="mean_squared_error": A função de perda (loss function) mede o erro entre o valor previsto pela rede e o valor real (y).
#        O erro quadrático médio (mean squared error, MSE) calcula a média dos quadrados das diferenças entre as previsões e os valores reais.
#        Usamos essa função porque ela penaliza erros grandes e é adequada para problemas de regressão (previsão de valores contínuos).

# Compilando o modelo
model.compile(optimizer="adam", loss="mean_squared_error")

# 4. Treinamento do modelo:
#    - model.fit(X, y, epochs=20): treina a rede neural usando os dados X e y por 20 épocas (passagens completas pelo conjunto de dados).
#      A cada época, o otimizador Adam ajusta os pesos da rede para tentar minimizar a função de perda (MSE), ou seja, para que as previsões
#      fiquem o mais próximas possível dos valores reais.

# Treinando o modelo
model.fit(X, y, epochs=20)

# 5. Avaliação do modelo:
#    - model.evaluate(X, y): avalia o desempenho do modelo nos dados de treinamento, retornando a perda (MSE) e outras métricas.
#      Isso ajuda a entender como o modelo está se saindo em relação aos dados que ele já viu.

loss = model.evaluate(X, y)
print("\nLoss (Mean Squared Error) nos dados de treinamento:", loss)

# 6. Teste do modelo em novos dados:
#    - Vamos criar um novo conjunto de dados X_teste com apenas 10 amostras.
#    - Calculamos y_teste usando a mesma função não linear.
#    - Fazemos a predição com o modelo treinado e comparamos com os valores reais.

X_teste = np.random.random((10, 3))
y_teste = evaluate_polynomial(X_teste)

# Predição do modelo
y_pred = model.predict(X_teste)

print("\nValores reais (y_teste):")
print(y_teste.flatten())
print("\nValores previstos pela rede neural:")
print(y_pred.flatten())


# 7. Plotando os dados de teste e as previsões em linhas com pyplot
import matplotlib.pyplot as plt

plt.plot(y_teste, label="Valores Reais")
plt.plot(y_pred, label="Valores Previstos")
plt.xlabel("Amostras")
plt.ylabel("Valores")
plt.legend()
plt.title("Comparação entre Valores Reais e Previstos")
plt.show()
