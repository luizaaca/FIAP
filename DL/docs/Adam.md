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
