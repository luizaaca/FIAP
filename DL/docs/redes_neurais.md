# Redes Neurais em Keras: Guia Didático

## 1. O que são redes neurais e para que servem?

Redes neurais artificiais são modelos computacionais inspirados no cérebro humano, compostos por unidades chamadas neurônios. Elas são capazes de aprender padrões complexos a partir de dados e são usadas para tarefas como classificação, regressão, reconhecimento de imagens, processamento de linguagem natural, entre outras.

- **Classificação:** Dizer se um e-mail é spam ou não, identificar dígitos em imagens, etc.
- **Regressão:** Prever preços de casas, temperatura, etc.
- **Outras aplicações:** Tradução automática, recomendação de produtos, detecção de fraudes, etc.

## 2. Como funcionam os neurônios artificiais?

Cada neurônio recebe vários valores de entrada (features), multiplica cada um por um peso, soma tudo e adiciona um termo chamado bias (viés):

    z = w1*x1 + w2*x2 + ... + wn*xn + b

- **x1...xn:** valores de entrada (ex: idade, renda, pixels de uma imagem)
- **w1...wn:** pesos do neurônio (ajustados durante o treinamento)
- **b:** bias (viés, também ajustado)

O resultado z é passado por uma função de ativação, que decide o que o neurônio "vai passar adiante".

### Exemplo prático (como em `rede_neural_simples_2.py`):

Suponha:
- w = [0.5, -0.2, 0.1, 0.3, -0.4, 0.7, 0.0, 0.2]
- b = 0.1
- x = [1.0, 0.5, 0.2, 0.0, 0.3, 0.7, 0.1, 0.4]

O cálculo é:

    z = (0.5*1.0) + (-0.2*0.5) + (0.1*0.2) + (0.3*0.0) + (-0.4*0.3) + (0.7*0.7) + (0.0*0.1) + (0.2*0.4) + 0.1 = 0.97

Depois, aplica-se a função de ativação.

## 3. Pesos e bias: o que são e para que servem?

- **Pesos:** Determinam a importância de cada entrada para o neurônio. São ajustados automaticamente durante o treinamento para que a rede aprenda a tarefa desejada.
- **Bias:** Permite que o neurônio "desloque" a função de ativação, ajudando a rede a se ajustar melhor aos dados.

## 4. Funções de ativação: para que servem e exemplos

As funções de ativação transformam o resultado z do neurônio, permitindo que a rede aprenda relações não lineares e tome decisões.

#### a) ReLU (Rectified Linear Unit)
- Fórmula: ReLU(z) = max(0, z)
- Se z > 0, passa o valor adiante; se z <= 0, vira zero.
- Muito usada em camadas ocultas.

#### b) Sigmoid
- Fórmula: sigmoid(z) = 1 / (1 + exp(-z))
- Resultado entre 0 e 1 (interpretação como probabilidade).
- Usada em saídas para classificação binária.

#### c) Softmax
- Fórmula: softmax(z_i) = exp(z_i) / sum_j exp(z_j) (para cada saída i)
- Transforma um vetor de valores em probabilidades que somam 1.
- Usada em saídas para classificação multiclasse.

#### d) Tanh (Tangente Hiperbólica)
- Fórmula: tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
- Resultado entre -1 e 1.
- Usada em algumas redes para normalizar saídas.

## 5. Como as camadas se conectam e como os dados passam

- Cada camada (**Dense** ou **Linear**) recebe como entrada o vetor de saídas da camada anterior.
- Todos os neurônios de uma camada recebem o mesmo vetor de entrada em paralelo.
- A saída de cada camada é uma matriz (número de exemplos, número de neurônios).
- O fluxo é sempre: entrada → camada 1 → camada 2 → ... → camada de saída.
- Não existe "passar de um neurônio para o outro" dentro da mesma camada.

### 5.1 Outros tipos de camadas além da Dense

Além das camadas **Dense** ou **Linear** (totalmente conectadas - W·x + b), existem outros tipos de camadas em redes neurais, cada uma adequada para diferentes tipos de dados e tarefas:

- **Dense ou Linear (Totalmente conectada):** Cada neurônio recebe todas as entradas da camada anterior. Usada em dados tabulares e como “cabeça” de redes para outras tarefas.
- **Conv2D (Convolucional):** Usada para processar imagens. Cada neurônio (filtro) analisa apenas uma pequena região da entrada de cada vez, aprendendo padrões espaciais.
- **LSTM, GRU, SimpleRNN (Recorrentes):** Usadas para sequências e séries temporais (texto, áudio, etc.), pois conseguem “lembrar” informações anteriores.
- **Dropout:** Não aprende padrões, mas serve para regularização, “desligando” aleatoriamente neurônios durante o treino.
- **Flatten, Reshape:** Apenas transformam o formato dos dados, sem aprender nada.
- **BatchNormalization:** Normaliza os dados dentro da rede para acelerar e estabilizar o treinamento.
- **Embedding:** Transforma índices inteiros (ex: palavras) em vetores densos, muito usada em NLP.

Cada tipo de camada tem um papel específico e pode ser combinada com outras para criar arquiteturas poderosas!

## 6. Modos de criar redes no Keras

#### a) API Sequencial
- Mais simples, camadas adicionadas uma após a outra.
- Exemplo:

```python
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(1, activation='sigmoid'))
```

#### b) API Funcional (Model)
- Mais flexível, permite múltiplas entradas/saídas, ramificações, etc.
- Exemplo:

```python
from keras import Input, Model
from keras.layers import Dense
inputs = Input(shape=(8,))
x = Dense(10, activation='relu')(inputs)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)
```

#### c) Subclassing (herança de Model)
- Para arquiteturas personalizadas e controle total do forward pass.
- Exemplo:

```python
from keras import Model
from keras.layers import Dense
class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(10, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
model = MyModel()
```

> Consulte os arquivos `rede_neural_simples.py` e `rede_neural_simples_2.py` para exemplos práticos e comentários didáticos.

## 7. Como criar redes no PyTorch

O PyTorch é uma biblioteca muito popular para deep learning, especialmente em pesquisa, pois oferece grande flexibilidade e controle manual sobre o treinamento.

#### a) Definindo a arquitetura (classe nn.Module)
- Você cria uma classe que herda de `nn.Module` e define as camadas no método `__init__`.
- O método `forward` define como os dados passam pelas camadas.

Exemplo:

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)  # Primeira camada
        self.fc2 = nn.Linear(10, 1)  # Segunda camada

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
```

#### b) Treinamento manual (loop de epochs)
- Você precisa definir a função de perda (loss), o otimizador e controlar manualmente cada etapa do treinamento.

Exemplo:

```python
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

> Veja o arquivo `rede_neural_simples_3.py` para um exemplo completo e comentado.

## 8. Como funciona o backpropagation (retropropagação) em Keras e PyTorch

O backpropagation é o algoritmo que calcula os gradientes dos pesos da rede neural para que o otimizador possa ajustá-los e melhorar o desempenho do modelo.

#### a) Keras (TensorFlow)
- O Keras (com TensorFlow) automatiza todo o processo de backpropagation.
- Ao chamar `model.fit(...)`, o framework constrói o grafo computacional, calcula os gradientes e atualiza os pesos automaticamente.
- Você raramente precisa lidar diretamente com gradientes ou grafos.
- Exemplo:
    ```python
    model.fit(X, y, epochs=100)
    ```
- O método `fit` cuida de tudo: forward, cálculo do erro, backpropagation e atualização dos pesos.

#### b) PyTorch
- No PyTorch, o processo é mais manual e explícito.
- Ao calcular a saída (`outputs = model(X)`), o PyTorch constrói dinamicamente o grafo computacional.
- O cálculo da perda (`loss = criterion(outputs, y)`) expande esse grafo até o tensor de saída.
- Ao chamar `loss.backward()`, o PyTorch percorre o grafo de trás para frente, calculando os gradientes de cada parâmetro.
- O otimizador (`optimizer.step()`) usa esses gradientes para atualizar os pesos.
- Você tem controle total sobre cada etapa, podendo customizar o processo se desejar.

### 8.1 A matemática do backpropagation

O backpropagation (retropropagação) é o algoritmo que permite que as redes neurais "aprendam" ajustando os pesos para reduzir o erro. Ele usa cálculo diferencial (derivadas) para descobrir como cada peso influencia o erro final.

Veja o passo a passo simplificado:

#### 1. Forward pass (passagem para frente)
- Os dados de entrada passam pela rede camada a camada.
- Cada neurônio faz o cálculo: `z = w1*x1 + w2*x2 + ... + wn*xn + b`.
- O resultado z passa pela função de ativação (ex: ReLU, sigmoid).
- No final, a rede gera uma previsão (output).

#### 2. Cálculo do erro (loss)
- Compara a previsão da rede com o valor real (target) usando uma função de perda (ex: MSE, cross-entropy).
- Exemplo: `erro = (y_pred - y_real)^2` (para regressão).

#### 3. Backward pass (retropropagação)
- O objetivo é descobrir como cada peso contribuiu para o erro.
- Usando a regra da cadeia da derivada, calcula-se o quanto o erro mudaria se cada peso mudasse um pouquinho.
- Para cada peso, calcula-se o gradiente: `d(erro)/d(peso)`.
- O cálculo começa do final da rede (saída) e vai "voltando" camada por camada até a entrada.

##### Exemplo didático (1 neurônio):
Suponha um neurônio com saída `y_pred = f(z)`, onde `z = w1*x1 + w2*x2 + b` e f é uma função de ativação.

1. Calcula o erro: `erro = (y_pred - y_real)^2`
2. Calcula a derivada do erro em relação à saída: `d(erro)/d(y_pred)`
3. Calcula a derivada da saída em relação a z: `d(y_pred)/d(z)` (depende da função de ativação)
4. Calcula a derivada de z em relação a cada peso: `d(z)/d(w1) = x1`, `d(z)/d(w2) = x2`
5. Multiplica tudo (regra da cadeia):
   - `d(erro)/d(w1) = d(erro)/d(y_pred) * d(y_pred)/d(z) * d(z)/d(w1)`
   - Isso diz quanto mudar w1 afeta o erro.

**Exemplo numérico simples:**

Suponha:
- x1 = 2
- w1 = 0.5
- b = 0
- Função de ativação: identidade (ou seja, y_pred = z)
- y_real = 5

1. Calcula a saída do neurônio:
    - z = w1 * x1 + b = 0.5 * 2 + 0 = 1
    - y_pred = z * 1 = 1
2. Calcula o erro (loss):
    - erro = (y_pred - y_real)² = (1 - 5)² = 16
3. Calcula a derivada (com regra da cadeia) do erro em relação ao peso w1:
    - d(erro)/d(w1) = d(erro)/d(y_pred) * d(y_pred)/d(z) * d(z)/d(w1)
        - d(z)/d(w1) = x1 = 2. Como z = w1 * x1 + b = w1¹ * x1 (constantes são desconsideradas), logo d(z) = d(w1¹ * x1), então = 1 * w1⁰ * x1 = x1.
        - d(y_pred)/d(z) = 1. Como y_pred = f(z) = z¹, d(y_pred) = d(z¹) = 1 * zº = 1.
        - d(erro)/d(y_pred) = d((y_pred - y_real)²) =  2 * (y_pred - y_real) = 2 * (1 - 5) = -8 (função de erro MSE)
    - d(erro)/d(w1) = -8 * 1 * 2 = -16

**OBS: df/dx onde '/' é notação, não é divisão**

Ou seja, o gradiente para w1 é -16. Isso indica que, para reduzir o erro, devemos aumentar o valor de w1 (pois o gradiente é negativo).

No exemplo do gradiente acima, usamos a regra da cadeia para combinar várias derivadas e descobrir como o erro muda em relação a cada peso.

#### 4. Atualização dos pesos
- O otimizador (ex: SGD, Adam) usa os gradientes para ajustar cada peso:
    - `novo_peso = peso_atual - taxa_aprendizado * gradiente`
- Assim, os pesos que mais aumentam o erro são mais corrigidos.

#### Resumindo:
1. Passa os dados pela rede (forward)
2. Calcula o erro
3. Calcula os gradientes de cada peso (backward)
4. Atualiza os pesos para reduzir o erro

Esse ciclo se repete por várias épocas até a rede aprender a tarefa!

#### Resumo comparativo
| Aspecto               | Keras (TensorFlow)               | PyTorch                          |
|-----------------------|----------------------------------|----------------------------------|
| Grafo computacional   | Estático (TF1) / Dinâmico (TF2)  | Dinâmico (define a cada forward) |
| Backpropagation       | Automático via `fit()`           | Manual via `loss.backward()`     |
| Atualização dos pesos | Automática via `fit()`           | Manual via `optimizer.step()`    |
| Flexibilidade         | Menos flexível, mais simples     | Mais flexível, mais controle     |

> Em ambos, o grafo computacional conecta a saída (loss) até os parâmetros do modelo, permitindo calcular os gradientes e atualizar os pesos automaticamente.
