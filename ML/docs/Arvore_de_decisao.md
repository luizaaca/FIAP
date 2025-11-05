## Árvore de Decisão — Explicação didática

Uma árvore de decisão é um modelo de machine learning para tarefas de classificação ou regressão que representa decisões em forma de árvore. Ela divide repetidamente o espaço de entrada em regiões mais homogêneas usando condições (splits) sobre as features.

- Estrutura básica:
	- Raiz (root): o nó inicial que contém todo o conjunto de dados.
	- Nós internos: representam perguntas/splits sobre uma variável (por exemplo, "idade <= 30?").
	- Folhas (leaves): representam a predição final — uma classe (classificação) ou um valor (regressão).

- Como funciona (ideia intuitiva):
	1. Escolhe-se uma feature e um ponto de corte que melhor separa os exemplos segundo um critério (por exemplo, Gini, Entropia/Information Gain, ou variância para regressão).
	2. Divide-se o conjunto em dois (ou mais) ramos com base nesse corte.
	3. Repete-se o processo recursivamente em cada ramo até um critério de parada (profundidade máxima, número mínimo de amostras, ou pureza da folha).

- Critérios de divisão comuns:
	- Entropia / Ganho de Informação (Information Gain): mede a redução de incerteza.
	- Gini impurity: frequência das classes incorretas — usado frequentemente em Random Forests (Scikit-learn usa Gini por padrão).
	- Variância (para regressão): escolhe cortes que minimizam a variância dentro das folhas.

- Hiperparâmetros importantes:
	- max_depth: profundidade máxima da árvore (controla overfitting).
	- min_samples_split: número mínimo de amostras para dividir um nó.
	- min_samples_leaf: número mínimo de amostras que uma folha deve ter.
	- max_features: número de features consideradas ao procurar o melhor split.
	- criterion: função usada para medir a qualidade do split ("gini", "entropy", "mse" para regressão).

- Overfitting e poda (pruning):
	Árvores de decisão têm alta capacidade de ajuste e podem facilmente overfitar (decorar ruído). Técnicas para controlar isso:
	- Limitar profundidade (max_depth).
	- Requerer um número mínimo de amostras por nó/folha.
	- Poda pós-treinamento (cost-complexity pruning em scikit-learn: ccp_alpha).

- Vantagens:
	- Intuitiva e fácil de interpretar (visualização da árvore).
	- Pouco pré-processamento (não precisa necessariamente normalizar features).
	- Funciona com dados numéricos e categóricos (com tratamentos).

- Desvantagens:
	- Tende a overfitar se não regularizada.
	- Sensível a pequenas variações nos dados (alta variância).
	- Árvores profundas podem ser instáveis e complexas de interpretar.

- Exemplo rápido (Scikit-learn — classificação) — ideia:

```python
from sklearn.tree import DecisionTreeClassifier

# X: features (n_samples, n_features), y: rótulos
clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=0)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
```

Você pode visualizar a árvore com `plot_tree` ou exportá-la para `graphviz` para entender as regras.

- Dicas práticas:
	- Sempre teste com validação cruzada ao ajustar hiperparâmetros.
	- Considere ensemble methods (Random Forest, Gradient Boosting) se precisar de melhor performance e menor variância.
	- Para datasets com muitas features, use `max_features` ou selecione features relevantes para evitar splits pouco informativos.

## Exemplo didático (texto)

Considere um problema simples de aprovação de empréstimo onde temos 6 candidatos com duas características observáveis:

- Candidato A: idade 25, renda 30k, aprovado: não
- Candidato B: idade 45, renda 80k, aprovado: sim
- Candidato C: idade 22, renda 25k, aprovado: não
- Candidato D: idade 35, renda 60k, aprovado: sim
- Candidato E: idade 50, renda 90k, aprovado: sim
- Candidato F: idade 28, renda 55k, aprovado: não

Passo a passo (como a árvore pode ser construída e usada):

1. Raiz — escolher o melhor split: suponha que o critério identifique "renda >= 50k" como melhor separador. A árvore divide os candidatos em dois grupos: renda < 50k (A, C) e renda >= 50k (B, D, E, F).
2. Nó esquerdo (renda < 50k) — todos os exemplos são "não"; tornamos esse nó uma folha com predição = "não".
3. Nó direito (renda >= 50k) — tem mistura de respostas (B, D, E são "sim"; F é "não"). A árvore procura novo split nesse ramo. Suponha que "idade >= 40" seja o próximo melhor corte.
   - Subnó "idade >= 40": contém B e E (ambos "sim") → folha com predição = "sim".
   - Subnó "idade < 40": contém D (35, sim) e F (28, não) → ainda misturado; a árvore pode parar (seminário mínimo) ou dividir novamente. Se não houver splits úteis restantes, pode escolher a classe majoritária (neste caso, empate ou critério de desempate) — assumamos que vira folha com predição = "sim" porque D é mais representativo.
4. Uso para predição: para um novo candidato G com renda 52k e idade 30, a árvore verifica a raiz (renda >= 50k? sim) vai para o nó direito; em seguida verifica idade >= 40? não → vai para subnó idade < 40 → prediz "sim".

Este exemplo ilustra como a árvore divide os dados em regiões, cria folhas quando os grupos ficam homogêneos e como regras simples (tests) conduzem a uma decisão interpretável.
