# ----------------------------------------------------------
# REDE NEURAL SIMPLES EM PYTORCH PARA REGRESSÃO
# ----------------------------------------------------------
# Este código constrói e treina uma rede neural simples usando PyTorch.
# O objetivo é prever um valor contínuo (regressão) a partir de exemplos com 3 características (features).
# Cada "neurônio" aprende a combinar essas informações para tentar prever o valor de saída, ajustando seus
# pesos automaticamente durante o treinamento.
# ----------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Geração dos dados de entrada
# X: matriz com 100 amostras e 3 características (features)
#    Cada linha de X é um exemplo com 3 informações diferentes (ex: tamanho, peso, idade).
# y: vetor de saída com 100 valores (um para cada amostra)
#    Cada valor de y é o alvo que queremos prever.
X = torch.randn(
    100, 3
)  # 100 exemplos, cada um com 3 características (valores aleatórios)
y = torch.randn(100, 1)  # 100 valores de saída (alvos), também aleatórios

# 2. Definição da arquitetura da rede neural
# Uma rede neural é composta por camadas de neurônios. Cada neurônio faz uma soma ponderada das entradas
# e aplica uma função de ativação.
#
# Exemplo de cálculo em um neurônio:
#   z = w1*x1 + w2*x2 + w3*x3 + b
#   - x1, x2, x3: valores de entrada
#   - w1, w2, w3: pesos do neurônio (ajustados durante o treino)
#   - b: bias (viés, também ajustado)
#
# A saída z pode ser passada por uma função de ativação, como ReLU (Rectified Linear Unit):
#   ReLU(z) = max(0, z)
#
# A rede abaixo tem:
# - Primeira camada: 10 neurônios, cada um recebendo os 3 valores de entrada
# - Segunda camada: 1 neurônio, recebendo os 10 valores da camada anterior


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Primeira camada totalmente conectada (Linear):
        # Cada um dos 10 neurônios recebe os 3 valores de entrada e calcula sua própria soma ponderada + bias
        self.fc1 = nn.Linear(3, 10)
        # Segunda camada: 1 neurônio recebe os 10 valores da camada anterior
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        # Passa os dados pela primeira camada e aplica a ativação ReLU
        # Exemplo: se x = [1.0, 0.5, -0.2], cada neurônio calcula z = w1*1.0 + w2*0.5 + w3*(-0.2) + b
        # Depois aplica ReLU: se z > 0, passa adiante; se z <= 0, vira zero
        x = torch.relu(self.fc1(x))
        # Passa pela segunda camada (sem ativação, pois é regressão)
        # O resultado é o valor previsto para cada exemplo
        x = self.fc2(x)
        return x


# 3. Instancia o modelo, define a função de perda e o otimizador
model = SimpleNN()

# Função de perda: erro quadrático médio (MSE)
# Mede o quão distantes, em média, as previsões estão dos valores reais
criterion = nn.MSELoss()

# Otimizador Adam: ajusta automaticamente os pesos e bias da rede para minimizar o erro
# O objeto 'optimizer' mantém uma referência direta aos parâmetros do modelo (pesos e bias),
# pois recebe model.parameters() na sua criação. Assim, qualquer atualização feita pelo otimizador
# afeta diretamente os parâmetros do modelo.
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Treinamento do modelo
# O treinamento ocorre em épocas (epochs): cada época é uma passagem completa pelos dados
#
# OBSERVAÇÃO:
# Em PyTorch, o treinamento é feito de forma manual, controlando cada etapa do processo (forward, cálculo
# do erro, backpropagation, atualização dos pesos).
# Isso se assemelha ao modo "subclassing" do Keras, onde você define explicitamente o forward pass e tem
# controle total sobre o fluxo de dados e o treinamento.
#
for epoch in range(100):
    optimizer.zero_grad()  # Zera os gradientes acumulados (importante para não somar gradientes de épocas anteriores)
    outputs = model(X)  # Faz a previsão para todos os exemplos (forward pass)
    loss = criterion(outputs, y)  # Calcula o erro entre as previsões e os valores reais
    # IMPORTANTE:
    # O objeto 'loss' não acessa diretamente os parâmetros do modelo.
    # Ao calcular outputs = model(X), o PyTorch constrói um grafo computacional
    # que conecta os tensores de entrada, os parâmetros do modelo (pesos e bias) e todas as operações realizadas.
    # O tensor 'outputs' mantém referência a esse grafo, e ao calcular 'loss', esse grafo é expandido até o tensor de saída.
    # Quando chamamos loss.backward(), o PyTorch percorre esse grafo de trás para frente, encontrando todos os parâmetros
    # do modelo que participaram do cálculo de outputs e, consequentemente, do loss.

    loss.backward()  # Calcula os gradientes dos pesos (backpropagation) e armazena em param.grad de cada parâmetro
    # 1. loss.backward():
    #    - Quando você chama loss.backward(), o PyTorch percorre automaticamente toda a rede (a partir do tensor 'loss'),
    #      seguindo as referências dos tensores e operações usadas para calcular o resultado (o chamado "computational graph").
    #    - Cada parâmetro do modelo (pesos e bias) é um objeto torch.nn.Parameter, que mantém internamente um atributo .grad.
    #    - O método backward() calcula o gradiente do erro em relação a cada parâmetro e armazena esse valor em param.grad.
    #    - Ou seja, cada camada (nn.Linear, etc.) e cada parâmetro do modelo já está registrado no grafo computacional e
    #      mantém referência ao seu próprio gradiente.
    #

    optimizer.step()  # Atualiza os pesos e bias da rede usando os gradientes armazenados
    # 2. optimizer.step():
    #    - O otimizador (Adam, SGD, etc.) foi criado com model.parameters(), ou seja, ele mantém uma lista de referências
    #      diretas para todos os parâmetros do modelo.
    #    - Ao chamar optimizer.step(), o otimizador percorre cada parâmetro, acessa o valor atual (param.data) e o gradiente
    #      calculado (param.grad), e faz a atualização dos pesos de acordo com a regra do otimizador.
    #    - Não é necessário passar parâmetros explicitamente, pois o otimizador já "sabe" quais parâmetros deve atualizar
    #      graças à referência mantida internamente.
    #
    # Resumindo: os objetos do modelo e do otimizador mantêm referências diretas aos parâmetros e seus gradientes,
    # permitindo que as funções loss.backward() e optimizer.step() atualizem os pesos automaticamente.
    # Assim, os gradientes são calculados para cada parâmetro envolvido, porque o grafo computacional liga:
    # loss → outputs → operações → parâmetros do modelo.
    #

    # Exibe o erro (loss) da época atual. Quanto menor, melhor o modelo está se ajustando aos dados.
    print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
