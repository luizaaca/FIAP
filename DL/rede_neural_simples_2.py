from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# EXPLICAÇÃO DETALHADA DO CÓDIGO
#
# Este código constrói e treina uma rede neural simples para classificação binária usando Keras.
# O objetivo é prever um valor binário (0 ou 1) a partir de 8 características de entrada.
# Esse tipo de rede é versátil para tarefas supervisionadas com dados tabulares, principalmente
# classificação binária, mas também pode ser adaptada para multiclasse e regressão. Não é
# indicada para imagens, texto sequencial ou dados com estrutura espacial/temporal complexa
# (para isso, use CNNs, RNNs, etc.).

# ----------------------------------------------------------
# COMO FUNCIONA UM NEURÔNIO DENSO COM ATIVAÇÃO RELU
#
# Cada neurônio denso recebe todos os valores de entrada (features) e faz uma soma ponderada:
#   z = w1*x1 + w2*x2 + ... + w8*x8 + b
#   - x1...x8: valores de entrada (ex: idade, renda, etc)
#   - w1...w8: pesos do neurônio (inicialmente aleatórios e ajustados durante o treinamento)
#   - b: bias (viés, também ajustado)
#
# A ativação ReLU (Rectified Linear Unit) transforma o resultado z:
#   ReLU(z) = max(0, z)
#   - Se z > 0, passa o valor adiante
#   - Se z <= 0, vira zero
#
# Exemplo prático:
#   Suponha que os pesos e bias de um neurônio sejam:
#     w = [0.5, -0.2, 0.1, 0.3, -0.4, 0.7, 0.0, 0.2]
#     b = 0.1
#     x = [1.0, 0.5, 0.2, 0.0, 0.3, 0.7, 0.1, 0.4]
#   O cálculo é:
#     z = (0.5*1.0) + (-0.2*0.5) + (0.1*0.2) + (0.3*0.0) + (-0.4*0.3) + (0.7*0.7) + (0.0*0.1) + (0.2*0.4) + 0.1
#     z = 0.97
#   Aplicando ReLU:
#     ReLU(0.97) = 0.97
#     Se z fosse negativo, o resultado seria 0.
#
# Durante o treinamento, os pesos (w) e bias (b) são ajustados automaticamente para que a rede aprenda a tarefa desejada.
# ----------------------------------------------------------

# ----------------------------------------------------------
# COMO FUNCIONA UM NEURÔNIO DENSO COM ATIVAÇÃO SIGMOID
#
# Um neurônio denso com ativação sigmoid também recebe todos os valores de entrada (features) e faz uma soma ponderada:
#   z = w1*x1 + w2*x2 + ... + wn*xn + b
#   - x1...xn: valores de entrada (podem ser as saídas da camada anterior)
#   - w1...wn: pesos do neurônio (ajustados durante o treinamento)
#   - b: bias (viés, também ajustado)
#
# A diferença está na função de ativação:
#   sigmoid(z) = 1 / (1 + exp(-z))
#   - O resultado da sigmoid é sempre um número entre 0 e 1.
#   - Pode ser interpretado como uma probabilidade.
#
# Exemplo prático:
#   Suponha que z = 2.0
#   sigmoid(2.0) = 1 / (1 + exp(-2.0)) ≈ 0.88
#   Ou seja, o neurônio "acredita" que a saída deve ser 0.88 (ou 88% de chance de ser classe 1).
#   Se z = -2.0, sigmoid(-2.0) ≈ 0.12 (12% de chance de ser classe 1).
#
# Durante o treinamento, os pesos (w) e bias (b) são ajustados para que a saída do neurônio se aproxime do valor real desejado (0 ou 1).
# ----------------------------------------------------------

# ----------------------------------------------------------
# COMO FUNCIONA A PASSAGEM DE INFORMAÇÕES NO MODELO SEQUENCIAL
#
# 1. Para cada exemplo do dataset (por exemplo, X[0]), todos os neurônios da primeira camada recebem os 8 valores de entrada ao mesmo tempo (em paralelo).
#    - Cada neurônio faz sua própria soma ponderada dos 8 valores e aplica a ativação (ReLU).
#    - O resultado é um vetor de 12 valores (um de cada neurônio) para cada exemplo.
#
# 2. A segunda camada recebe, para cada exemplo, esse vetor de 12 valores (saída da primeira camada).
#    - Cada neurônio da segunda camada faz uma soma ponderada desses 12 valores e aplica a ativação (ReLU).
#    - O resultado é um vetor de 8 valores (um de cada neurônio da segunda camada) para cada exemplo.
#
# 3. A camada de saída recebe, para cada exemplo, os 8 valores da segunda camada.
#    - O único neurônio da saída faz uma soma ponderada desses 8 valores, aplica a ativação sigmoid e gera um número entre 0 e 1 (probabilidade).
#
# 4. Isso acontece para todos os exemplos do dataset, um por vez (mas pode ser processado em lotes para acelerar).
#
# O que é processamento em lotes (batch)?
# - Em vez de passar um exemplo de cada vez pela rede, o modelo pode processar vários exemplos juntos, formando um "lote" (batch).
# - Por exemplo, se batch_size=10, a rede processa 10 exemplos ao mesmo tempo em cada etapa.
#
# O que muda?
# - O cálculo interno de cada camada é feito usando matrizes, aproveitando operações vetorizadas (mais rápidas em hardware moderno).
# - O gradiente (ajuste dos pesos) é calculado com base na média do erro de todos os exemplos do lote, não de um só.
# - Isso acelera o treinamento e pode ajudar a rede a generalizar melhor, pois o ajuste dos pesos considera vários exemplos de uma vez.
#
# Por que usar lotes?
# - Eficiência: Computadores (especialmente GPUs) são muito mais rápidos processando matrizes do que exemplos individuais.
# - Estabilidade: O ajuste dos pesos fica menos "ruidoso", pois considera a média do erro de vários exemplos.
# - Controle: O parâmetro batch_size permite equilibrar uso de memória e velocidade.
#
# Resumindo:
# - Com batch_size=1: processamento é totalmente sequencial (um exemplo por vez).
# - Com batch_size maior: processamento é paralelo dentro do lote, acelerando o treinamento e tornando o ajuste dos pesos mais estável.
# - O valor padrão da Keras é 32, mas pode ser ajustado conforme o problema e os recursos disponíveis.
#
# Resumindo:
# - Não existe "passar de um neurônio para o outro" dentro da mesma camada. Todos os neurônios de uma camada recebem o mesmo exemplo em paralelo.
# - Cada camada transforma a dimensão dos dados:
#     - Entrada: (100, 8)
#     - Após 1ª camada: (100, 12)
#     - Após 2ª camada: (100, 8)
#     - Saída final: (100, 1)
# - O fluxo é sempre: camada → próxima camada, nunca "dentro" da mesma camada.
# ----------------------------------------------------------

# Exemplo de dados de entrada
# X: matriz com 100 amostras e 8 características (features)
#    Cada linha de X representa um exemplo com 8 informações diferentes.
# y: vetor de saída com 100 valores binários (0 ou 1)
#    Cada valor de y indica a classe do exemplo correspondente em X.

# Gerando X normalmente
X = np.random.random((100, 8))


# Função para gerar y binário a partir de X
# Exemplo: soma ponderada das features + threshold
def generate_binary_labels(X):
    # Soma ponderada simples (pode ser ajustada)
    s = (
        2 * X[:, 0]
        + 1.5 * X[:, 1]
        - 1 * X[:, 2]
        + 0.5 * X[:, 3]
        + X[:, 4]
        - 0.7 * X[:, 5]
        + 0.3 * X[:, 6]
        - 1.2 * X[:, 7]
    )
    # Aplica threshold: se s > 0, classe 1; senão, classe 0
    return (s > 0).astype(int).reshape(-1, 1)


y = generate_binary_labels(X)


# Definindo o modelo sequencial (camada a camada)
# ----------------------------------------------------------
# O modelo é do tipo 'Sequencial', ou seja, as informações passam por uma camada de cada vez, em sequência.
# Pense como uma linha de montagem: cada camada faz um processamento e passa para a próxima.
model = Sequential()

# Primeira camada densa (oculta) com 12 neurônios, ativação ReLU, esperando 8 entradas (input_dim=8)
# - 'Dense' significa que cada neurônio dessa camada recebe todas as 8 informações de entrada.
# - '12 neurônios': são 12 mini-unidades de processamento, cada uma tentando aprender um padrão diferente nos dados.
# - 'input_dim=8': diz que cada exemplo tem 8 características (ex: idade, renda, etc).
# - 'activation="relu"': cada neurônio só "passa adiante" valores positivos (se o resultado for negativo, vira zero).
#   Isso ajuda a rede a aprender relações mais complexas e não lineares.
model.add(Dense(12, input_dim=8, activation="relu"))

# Segunda camada oculta com 8 neurônios, ativação ReLU
# - Mais uma camada de processamento, agora com 8 neurônios.
# - Cada neurônio recebe informações de todos os 12 neurônios da camada anterior.
# - Continua usando a ativação 'relu' para manter a capacidade de aprender padrões complexos.
model.add(Dense(8, activation="relu"))

# Camada de saída com 1 neurônio e ativação sigmoid (para classificação binária)
# - Só tem 1 neurônio porque queremos prever um único valor: a probabilidade de ser classe 1 (ex: "sim" ou "não").
# - 'activation="sigmoid"': transforma o resultado em um número entre 0 e 1, que pode ser interpretado como probabilidade.
#   Se o valor for maior que 0.5, normalmente dizemos que é classe 1; se for menor, é classe 0.
# - Exemplo: se a saída for 0.8, significa que o modelo acha que há 80% de chance de ser classe 1.
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

# Testando o modelo com novos dados
X_novo = np.random.random((5, 8))  # 5 novos exemplos
y_calculado = generate_binary_labels(X_novo)
y_pred = model.predict(X_novo)
print("Novos exemplos:\n", pd.DataFrame(X_novo))
print("Valores reais calculados:\n", pd.DataFrame(y_calculado))
print("Predições do modelo:\n", pd.DataFrame(y_pred))
