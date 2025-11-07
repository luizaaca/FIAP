# SGD (Stochastic Gradient Descent) – Explicação Didática e Exemplo Simples

## O que é o SGD?

SGD (Stochastic Gradient Descent) é um método usado para treinar redes neurais e outros modelos de aprendizado de máquina. Ele serve para encontrar os melhores valores para os pesos do modelo, ou seja, aqueles que minimizam o erro nas previsões.

## Como funciona o SGD?

1. **Escolhe um exemplo dos dados** (ou um pequeno grupo, chamado de "mini-batch").
2. **Calcula o erro** do modelo para esse exemplo.
3. **Calcula o gradiente** (quanto cada peso contribui para o erro).
4. **Atualiza os pesos** na direção que diminui o erro.
5. **Repete** esse processo para vários exemplos, várias vezes (épocas).

## Exemplo passo a passo

Imagine que queremos ajustar o peso w para prever o valor y a partir de x usando a fórmula:

    y_pred = w * x

Suponha que temos um único dado:
- x = 2
- y_real = 5
- Começamos com w = 1

### 1. Calcula a previsão
    y_pred = w * x = 1 * 2 = 2

### 2. Calcula o erro (usando erro quadrático)
    erro = (y_pred - y_real)^2 = (2 - 5)^2 = 9

### 3. Calcula o gradiente do erro em relação a w
    d(erro)/d(w) = 2 * (y_pred - y_real) * x
    d(erro)/d(w) = 2 * (2 - 5) * 2 = 2 * (-3) * 2 = -12

### 4. Atualiza o peso w
    w_novo = w - taxa_aprendizado * gradiente
    Suponha taxa_aprendizado = 0.1
    w_novo = 1 - 0.1 * (-12) = 1 + 1.2 = 2.2

### 5. Repete o processo
- Agora, com w = 2.2, repete os passos acima para o próximo exemplo ou para o mesmo exemplo em outra época.

## Resumo
- O SGD ajusta os pesos do modelo pouco a pouco, usando exemplos individuais ou pequenos grupos.
- Cada atualização é rápida e simples, tornando o treinamento eficiente.
- O processo é repetido até que o erro fique pequeno e o modelo aprenda a prever corretamente.

## Vantagens
- Simples de entender e implementar.
- Funciona bem para grandes conjuntos de dados.
- Pode escapar de mínimos locais (soluções ruins) por causa da aleatoriedade.

## Observação
- A taxa de aprendizado controla o tamanho do passo dado em cada atualização. Se for muito grande, pode não aprender; se for muito pequena, pode demorar muito.
