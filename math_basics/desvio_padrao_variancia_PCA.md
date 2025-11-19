# Desvio padrão, variância e relação com PCA (resumo didático)

Este resumo amarra os conceitos de variância e desvio padrão, e como eles se conectam ao PCA (Análise de Componentes Principais), aos autovetores e autovalores.

## Legenda dos símbolos

| Símbolo | Significado |
|---------|-------------|
| N       | Tamanho da população (número total de observações) |
| n       | Tamanho da amostra |
| xᵢ      | i‑ésima observação (valor individual) |
| Σ       | Somatório (soma sobre todas as observações) |
| μ       | Média populacional (parâmetro verdadeiro) |
| x̄       | Média amostral (estimativa de μ) |
| σ²      | Variância populacional |
| σ       | Desvio padrão populacional (√σ²) |
| s²      | Variância amostral (usa divisor n−1) |
| s       | Desvio padrão amostral (√s²) |
| λᵢ      | Autovalor i‑ésimo da matriz de covariância (variância explicada pelo componente i) |
| ratioᵢ  | Fração de variância explicada pelo componente i: λᵢ / Σ λⱼ |

## Conceitos básicos

- Média (centro): a “posição média” dos dados.
- Variância (espalhamento ao quadrado): mede o quão espalhados os dados estão em torno da média.
  - População: σ² = (1/N) · Σ (xi − μ)²
  - Amostra: s² = (1/(n−1)) · Σ (xi − x̄ )²
- Desvio padrão (espalhamento na mesma unidade dos dados): σ = √σ² ou s = √s².
  - Intuição: “distância típica” até a média. Pequeno = pontos concentrados; grande = pontos espalhados.

## Exemplo numérico simples

Dados: 2, 5, 8
- Média: (2+5+8)/3 = 5
- Desvios: −3, 0, +3
- Quadrados: 9, 0, 9 (soma = 18)
- Variância populacional: 18/3 = 6 → Desvio padrão populacional: √6 ≈ 2,45
- Variância amostral: 18/(3−1) = 9 → Desvio padrão amostral: √9 = 3

## Como isso se conecta ao PCA

PCA transforma os dados para um novo sistema de eixos (componentes principais) que capturam ao máximo a variação dos dados.

Passos essenciais (dados numéricos):
1. Centralizar: subtrair a média de cada coluna (o “centro” passa a ser 0).
2. Matriz de covariância: mede variâncias e covariâncias entre variáveis.
3. Autovetores (eixos novos) e autovalores (importância de cada eixo):
   - Autovetor = direção de maior variação (um “eixo” no espaço). Os componentes do PCA são autovetores da matriz de covariância.
   - Autovalor = variância dos dados projetados naquela direção.
   - Desvio padrão no componente = √(autovalor).
4. Ordenação: componentes são ordenados do maior para o menor autovalor (maior variância → mais “informação”).
5. Variância explicada: ratio_i = λ_i / Σ λ_j; acumulando esses ratios decidimos quantos componentes manter.

## Duas visões equivalentes do primeiro componente
- Maximizar variância: escolher a direção em que a variância projetada é máxima.
- Minimizar erro ortogonal de reconstrução: escolher a direção que minimiza a soma das distâncias perpendiculares (quadráticas) dos pontos até a reta/subespaço escolhido.
Essas formulações levam ao mesmo autovetor principal.

## O que NÃO confundir
- Autovetores não são médias. A média só serve para centralizar; depois, os autovetores descrevem direções de espalhamento.
- Autovalor não é “distância de um ponto até a reta”. Ele é a variância (coletiva) ao longo daquela direção; o desvio padrão nessa direção é sua raiz quadrada.
- O segundo componente é ortogonal (perpendicular) ao primeiro por construção; cada novo componente é ortogonal aos anteriores.

## Takeaways
- Variância e desvio padrão quantificam espalhamento em torno da média.
- No PCA, cada componente principal tem:
  - Uma direção (autovetor) e
  - Um espalhamento associado (autovalor = variância; desvio padrão = raiz do autovalor).
- Manter os componentes com maiores autovalores preserva mais da “informação” (variância) dos dados.


