## K-Means — Explicação didática

K-Means é um algoritmo de aprendizagem não supervisionada usado para agrupar (clusterizar) dados em k grupos com base na similaridade (distância). O objetivo é minimizar a soma das distâncias quadráticas (inércia) entre pontos e o centróide do seu cluster.

- Ideia geral:
    - Partitionamento: divide os dados em k clusters, cada um representado pelo seu centróide.
    - Cada ponto é atribuído ao cluster cujo centróide está mais próximo (normalmente distância Euclidiana).
    - Itera atribuição e recomputação de centróides até convergência.

- Passos do algoritmo (intuitivo):
    1. Inicializar k centróides (aleatório ou k-means++).
    2. Atribuir cada ponto ao centróide mais próximo.
    3. Calcular novos centróides como a média dos pontos atribuídos.
    4. Repetir passos 2–3 até que as atribuições não mudem ou atinja max_iter.

- Inicialização:
    - k-means++: estratégia recomendada para escolher centróides iniciais de forma que melhore convergência e reduz chute de resultados ruins.
    - Importante executar várias inicializações (n_init) e escolher a melhor pelo menor valor de inércia.

- Hiperparâmetros importantes:
    - n_clusters (k): número de clusters.
    - init: método de inicialização ('k-means++', 'random'...).
    - n_init: número de reinícios independentes.
    - max_iter: iterações máximas por execução.
    - tol: tolerância para critério de convergência.
    - random_state: para reprodutibilidade.

- Métricas e diagnóstico:
    - Inertia (soma das distâncias ao quadrado até centróide): objetivo interno, decresce com k.
    - Elbow method: plotar inertia vs k para escolher ponto de inflexão.
    - Silhouette score: mede quão bem os pontos se encaixam no seu cluster (melhor para comparar k).
    - Gap statistic: alternativa mais robusta para escolher k.

- Vantagens:
    - Rápido e simples, eficiente em muitas aplicações.
    - Fácil de interpretar centróides.
    - Escalável (MiniBatchKMeans para grandes datasets).

- Desvantagens e limitações:
    - Precisa definir k a priori.
    - Pressupõe clusters convexos e de formato aproximadamente esférico.
    - Sensível à escala das features (necessita normalização).
    - Sensível a outliers e inicialização (use n_init e k-means++).
    - Pode produzir clusters vazios (tratar casos).

- Dicas práticas:
    - Padronize/normalize as features (StandardScaler).
    - Teste diferentes k e use silhouette/elbow/gap para decisão.
    - Use k-means++ e aumente n_init (ex.: 10 ou mais).
    - Para grandes volumes, use MiniBatchKMeans.
    - Considere outras técnicas (DBSCAN, GaussianMixture) se clusters não forem esféricos ou densos.

- Exemplo rápido (Scikit-learn — clustering):

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline

# Pipeline com padronização
pipeline = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
)

labels = pipeline.fit_predict(X)
# Avaliação interna
score = silhouette_score(StandardScaler().fit_transform(X), labels)
```

- Alternativa para grandes datasets:
    - sklearn.cluster.MiniBatchKMeans: semelhante, mas atualiza centróides por mini-batches, reduz custo computacional.
## Exemplo didático (texto)

Considere seis pontos em 2D (coordenadas):

- P1 = (1, 1)
- P2 = (2, 1)
- P3 = (1, 2)
- P4 = (8, 8)
- P5 = (9, 8)
- P6 = (8, 9)

Objetivo: agrupar em k = 2 clusters.

1. Inicialização: suponha que os dois centróides iniciais escolhidos sejam C1 = (1,1) e C2 = (8,8).
2. Atribuição: para cada ponto, calcula-se a distância ao centróide mais próximo e atribui-se ao cluster correspondente.
    - P1, P2, P3 ficam próximos de C1 → cluster 1.
    - P4, P5, P6 ficam próximos de C2 → cluster 2.
3. Recalcular centróides: calcula-se a média dos pontos em cada cluster.
    - Novo C1 = média(P1,P2,P3) = ((1+2+1)/3, (1+1+2)/3) = (1.33, 1.33).
    - Novo C2 = média(P4,P5,P6) = ((8+9+8)/3, (8+8+9)/3) = (8.33, 8.33).
4. Repetir atribuição: com os novos centróides, as atribuições não mudam (cada ponto ainda fica com o centróide mais próximo).
5. Convergência: o algoritmo para; resultado final são os dois clusters descritos acima, com centróides aproximadamente (1.33, 1.33) e (8.33, 8.33).

Este exemplo mostra como K-Means separa naturalmente pontos próximos em grupos e ajusta centróides pela média até estabilizar as atribuições.