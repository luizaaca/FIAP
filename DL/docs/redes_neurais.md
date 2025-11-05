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

### a) ReLU (Rectified Linear Unit)
- Fórmula: ReLU(z) = max(0, z)
- Se z > 0, passa o valor adiante; se z <= 0, vira zero.
- Muito usada em camadas ocultas.

### b) Sigmoid
- Fórmula: sigmoid(z) = 1 / (1 + exp(-z))
- Resultado entre 0 e 1 (interpretação como probabilidade).
- Usada em saídas para classificação binária.

### c) Softmax
- Fórmula: softmax(z_i) = exp(z_i) / sum_j exp(z_j) (para cada saída i)
- Transforma um vetor de valores em probabilidades que somam 1.
- Usada em saídas para classificação multiclasse.

### d) Tanh (Tangente Hiperbólica)
- Fórmula: tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
- Resultado entre -1 e 1.
- Usada em algumas redes para normalizar saídas.

## 5. Como as camadas se conectam e como os dados passam

- Cada camada (**Dense**) recebe como entrada o vetor de saídas da camada anterior.
- Todos os neurônios de uma camada recebem o mesmo vetor de entrada em paralelo.
- A saída de cada camada é uma matriz (número de exemplos, número de neurônios).
- O fluxo é sempre: entrada → camada 1 → camada 2 → ... → camada de saída.
- Não existe "passar de um neurônio para o outro" dentro da mesma camada.

### 5.1 Outros tipos de camadas além da Dense

Além das camadas Dense (totalmente conectadas), existem outros tipos de camadas em redes neurais, cada uma adequada para diferentes tipos de dados e tarefas:

- **Dense (Totalmente conectada):** Cada neurônio recebe todas as entradas da camada anterior. Usada em dados tabulares e como “cabeça” de redes para outras tarefas.
- **Conv2D (Convolucional):** Usada para processar imagens. Cada neurônio (filtro) analisa apenas uma pequena região da entrada de cada vez, aprendendo padrões espaciais.
- **LSTM, GRU, SimpleRNN (Recorrentes):** Usadas para sequências e séries temporais (texto, áudio, etc.), pois conseguem “lembrar” informações anteriores.
- **Dropout:** Não aprende padrões, mas serve para regularização, “desligando” aleatoriamente neurônios durante o treino.
- **Flatten, Reshape:** Apenas transformam o formato dos dados, sem aprender nada.
- **BatchNormalization:** Normaliza os dados dentro da rede para acelerar e estabilizar o treinamento.
- **Embedding:** Transforma índices inteiros (ex: palavras) em vetores densos, muito usada em NLP.

Cada tipo de camada tem um papel específico e pode ser combinada com outras para criar arquiteturas poderosas!

## 6. Modos de criar redes no Keras

### a) API Sequencial
- Mais simples, camadas adicionadas uma após a outra.
- Exemplo:

```python
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(1, activation='sigmoid'))
```

### b) API Funcional (Model)
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

### c) Subclassing (herança de Model)
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

---

## Checklist de aprendizado

- [x] Entender o que são redes neurais e para que servem
- [x] Compreender o funcionamento dos neurônios artificiais
- [x] Saber o papel dos pesos e bias
- [x] Conhecer as principais funções de ativação (relu, sigmoid, softmax, tanh)
- [x] Entender como as camadas se conectam e como os dados fluem
- [x] Saber criar redes no Keras usando as três principais APIs

> Consulte os arquivos `rede_neural_simples.py` e `rede_neural_simples_2.py` para exemplos práticos e comentários didáticos.
