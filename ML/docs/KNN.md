````markdown
## K-Nearest Neighbors (KNN) — Explicação didática
K-Nearest Neighbors (KNN) é um algoritmo de aprendizado supervisionado usado para classificação e regressão. Ele baseia suas previsões na similaridade entre exemplos — para um novo ponto, consulta os k vizinhos mais próximos no espaço de features e decide a predição a partir desses vizinhos.
- Ideia geral:
	- Não há fase explícita de treinamento além de armazenar os dados de treinamento.
	- A predição é feita consultando os k pontos de treino mais próximos segundo uma métrica de distância.
- Como funciona (ideia intuitiva):
	1. Escolhe-se um valor para k (número de vizinhos) e uma métrica de distância (por exemplo, Euclidiana).
	2. Para cada novo exemplo, calcula-se a distância entre esse exemplo e todos os pontos de treino.
	3. Selecionam-se os k pontos mais próximos (menores distâncias).
	4. Classificação: a classe é definida por voto majoritário entre os vizinhos (ou voto ponderado por distância). Regressão: utiliza-se a média (ou média ponderada) dos valores dos vizinhos.
- Métricas de distância comuns:
	- Distância Euclidiana (L2): padrão para dados contínuos.
	- Distância Manhattan (L1).
	- Distância de Minkowski (generaliza L1 e L2).
	- Distâncias para dados categóricos (Hamming) ou medidas customizadas.
- Hiperparâmetros importantes:
	- k: número de vizinhos (controla viés/variância).
	- weights: tipo de ponderação ("uniform" ou "distance").
	- metric: função de distância usada.
	- algorithm: método para busca dos vizinhos ("auto", "ball_tree", "kd_tree", "brute").
	- leaf_size: parâmetro para árvores (afeta performance de consultas).
- Vantagens:
	- Simples de entender e implementar.
	- Não requer etapa de treinamento complexa.
	- Funciona bem com dados multimodais e decisões locais.
- Desvantagens:
	- Custo de predição alto para grandes datasets (busca linear sem estruturas).
	- Sensível à escala das features — requer normalização/standardization.
	- Escolha de k e da métrica pode impactar fortemente o desempenho.
	- Sofre com a "maldição da dimensionalidade" em espaços de alta dimensão.
- Dicas práticas:
	- Sempre normalize ou padronize as features quando usar distâncias.
	- Escolha k através de validação cruzada (evitar k muito baixo ou muito alto).
	- Considere ponderação por distância se vizinhos mais próximos forem mais confiáveis.
	- Use estruturas de índice (KD-Tree, Ball-Tree) ou aproximações para acelerar consultas em datasets grandes.
	- Para alta dimensionalidade, avalie redução de dimensionalidade (PCA, seleção de features).
- Exemplo rápido (Scikit-learn — classificação) — ideia:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# X: features (n_samples, n_features), y: rótulos
clf = make_pipeline(
	StandardScaler(),
	KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)
)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
```

## Exemplo didático (texto)

Suponha seis pontos em um plano 2D com classes ``R`` (vermelho) e ``B`` (azul):

- A = (0,0) → R
- B = (0,1) → R
- C = (1,0) → R
- D = (5,5) → B
- E = (5,6) → B
- F = (6,5) → B

Queremos classificar o ponto novo X = (1,1) usando k = 3 e distância Euclidiana.

1. Calcula-se a distância de X a cada ponto de treino:
	- d(X,A) = distância entre (1,1) e (0,0) ≈ 1.41
	- d(X,B) = distância entre (1,1) e (0,1) = 1.00
	- d(X,C) = distância entre (1,1) e (1,0) = 1.00
	- d(X,D) ≈ 5.66, d(X,E) ≈ 5.83, d(X,F) ≈ 5.66
2. Ordena-se por distância e toma-se os 3 mais próximos: B (1.00, R), C (1.00, R), A (1.41, R).
3. Votação majoritária entre os vizinhos: R, R, R → predição final = R.

Conclusão: com k=3, os vizinhos mais próximos de X são todos da classe vermelha, logo KNN classifica X como R. Este exemplo ilustra cálculo de distâncias, seleção dos k vizinhos e votação majoritária sem necessidade de treino explícito.
