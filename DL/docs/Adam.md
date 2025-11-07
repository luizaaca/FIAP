# Adam – Explicação Didática e Comparação com SGD

## O que é o Adam?

Adam (Adaptive Moment Estimation) é um método avançado para treinar redes neurais e outros modelos de aprendizado de máquina. Ele é uma evolução do SGD, trazendo melhorias que tornam o treinamento mais rápido e estável.

## Como funciona o Adam?

Adam parte do mesmo princípio do SGD (veja o arquivo `SGD.md` para entender o básico), mas faz duas coisas extras:

1. **Guarda o histórico dos gradientes:**
   - Adam calcula a média dos gradientes anteriores (chamado de "momento") e também a média dos quadrados dos gradientes (para saber se o gradiente está mudando muito).
2. **Ajusta o passo de cada peso automaticamente:**
   - Em vez de usar sempre o mesmo tamanho de passo (taxa de aprendizado), Adam ajusta o passo para cada peso, dependendo do histórico dos gradientes.

## Passo a passo simplificado


Imagine que você está ajustando o peso w para prever y a partir de x, como no exemplo do SGD:

  y_pred = w * x

Adam faz:
- Calcula o gradiente do erro em relação a w (igual ao SGD).
- Atualiza dois "acumuladores":
  - m (média dos gradientes)
  - v (média dos quadrados dos gradientes)
- Usa m e v para ajustar o tamanho do passo de atualização de w.

## Exemplo prático com valores (continuação do exemplo do SGD)

Vamos usar os mesmos valores do exemplo do SGD:
- x = 2
- y_real = 5
- w = 1
- taxa_aprendizado = 0.1
- m = 0 (inicialmente)
- v = 0 (inicialmente)
- beta1 = 0.9 (valor padrão)
- beta2 = 0.999 (valor padrão)
- epsilon = 1e-8 (valor padrão)

### 1. Calcula a previsão
  y_pred = w * x = 1 * 2 = 2

### 2. Calcula o erro
  erro = (y_pred - y_real)^2 = (2 - 5)^2 = 9

### 3. Calcula o gradiente do erro em relação a w
  gradiente = 2 * (y_pred - y_real) * x = 2 * (2 - 5) * 2 = -12

### 4. Atualiza os acumuladores m e v
  m = beta1 * m + (1 - beta1) * gradiente
    = 0.9 * 0 + 0.1 * (-12) = -1.2
  v = beta2 * v + (1 - beta2) * (gradiente^2)
    = 0.999 * 0 + 0.001 * (144) = 0.144

### 5. Corrige o viés dos acumuladores (primeira iteração)
  m_corrigido = m / (1 - beta1^1) = -1.2 / (1 - 0.9) = -12
  v_corrigido = v / (1 - beta2^1) = 0.144 / (1 - 0.999) = 0.144 / 0.001 = 144

### 6. Atualiza o peso w
  w_novo = w - taxa_aprendizado * m_corrigido / (sqrt(v_corrigido) + epsilon)
       = 1 - 0.1 * (-12) / (sqrt(144) + 1e-8)
       = 1 - 0.1 * (-12) / 12
       = 1 - 0.1 * (-1)
       = 1 + 0.1 = 1.1

Portanto, após uma iteração do Adam, o novo valor de w seria 1.1 (enquanto no SGD foi 2.2).

Se repetir o processo, os acumuladores m e v vão sendo atualizados e o Adam vai ajustando o passo automaticamente.

### Fórmulas principais (simplificadas)
- m = média dos gradientes (momento)
- v = média dos quadrados dos gradientes
- w_novo = w - taxa_aprendizado * m / (sqrt(v) + epsilon)
  - epsilon é um número pequeno para evitar divisão por zero

## Exemplo intuitivo

Suponha que o gradiente está mudando muito de um passo para outro:
- Adam percebe isso e diminui o passo para evitar "pular" demais.
Se o gradiente está estável:
- Adam aumenta o passo para aprender mais rápido.

## Vantagens do Adam
- Treina mais rápido que o SGD na maioria dos casos.
- Se adapta automaticamente ao comportamento dos gradientes.
- Funciona bem mesmo quando os dados são ruidosos ou o erro muda muito.

## Resumo
- Adam é como o SGD, mas mais inteligente: ele observa o histórico dos gradientes e ajusta o passo de cada peso automaticamente.
- Isso faz com que o treinamento seja mais eficiente e estável.

## Referência
Para entender o funcionamento básico do SGD, veja o arquivo [`SGD.md`](SGD.md).
