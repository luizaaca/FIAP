from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# ----------------------------------------------------------
# EXPLICAÇÃO DETALHADA DO CÓDIGO
#
# Este código constrói e treina uma rede neural simples para classificação binária usando Keras.
# O objetivo é prever um valor binário (0 ou 1) a partir de 8 características de entrada.
# ----------------------------------------------------------

# Exemplo de dados de entrada
# X: matriz com 100 amostras e 8 características (features)
#    Cada linha de X representa um exemplo com 8 informações diferentes.
# y: vetor de saída com 100 valores binários (0 ou 1)
#    Cada valor de y indica a classe do exemplo correspondente em X.
X = np.random.random((100, 8))
y = np.random.randint(2, size=(100, 1))

# Definindo o modelo sequencial (camada a camada)
model = Sequential()
# Primeira camada densa (oculta) com 12 neurônios, ativação ReLU, esperando 8 entradas (input_dim=8)
model.add(Dense(12, input_dim=8, activation="relu"))
# Segunda camada oculta com 8 neurônios, ativação ReLU
model.add(Dense(8, activation="relu"))
# Camada de saída com 1 neurônio e ativação sigmoid (para classificação binária)
model.add(Dense(1, activation="sigmoid"))

# Compilando o modelo
# - loss="binary_crossentropy": função de perda adequada para classificação binária
# - optimizer="adam": algoritmo de otimização eficiente e popular
# - metrics=["accuracy"]: mede a acurácia durante o treinamento
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinando o modelo
# - epochs=150: número de vezes que o modelo verá todo o conjunto de dados
# - batch_size=10: número de exemplos processados antes de atualizar os pesos
model.fit(X, y, epochs=150, batch_size=10)
