from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris

# Carrega o dataset Iris (flores), já separado em X (características) e y (rótulos)
data = load_iris(return_X_y=True)
X, y = data
print("Dados carregados. X shape:", X.shape, ", y shape:", y.shape)  # type: ignore

# Divide os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Dados divididos em treino e teste.")
print("X_train shape:", X_train.shape, ", X_test shape:", X_test.shape)

# Define o pipeline de processamento e classificação:
# 1. StandardScaler: padroniza os dados (média 0, desvio 1)
# 2. PCA: reduz a dimensionalidade para 3 componentes principais
# 3. RandomForestClassifier: classificador de floresta aleatória
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=3)),
        ("classifier", RandomForestClassifier()),
    ]
)
print("Pipeline criado com etapas: scaler -> pca -> classifier")

# Define a grade de parâmetros para o GridSearchCV
# Aqui, criamos um dicionário onde cada chave representa um hiperparâmetro do classificador dentro do pipeline.
# O prefixo 'classifier__' indica que o parâmetro pertence à etapa 'classifier' do pipeline.
# Para cada hiperparâmetro, fornecemos uma lista de valores possíveis.
# O GridSearchCV irá testar todas as combinações possíveis desses valores para encontrar a melhor configuração.
param_grid = {
    "classifier__n_estimators": [50, 100, 150],  # Quantidade de árvores na floresta
    "classifier__max_depth": [10, 20, 30],  # Profundidade máxima de cada árvore
}
print("Grade de parâmetros definida para GridSearchCV:")
print(param_grid)

# Executa o GridSearchCV para encontrar os melhores parâmetros do classificador
print("Executando GridSearchCV (isso pode demorar um pouco)...")
# O parâmetro cv=5 indica que será usada validação cruzada com 5 "folds".
# Isso significa que os dados de treino serão divididos em 5 partes:
# - Em cada rodada, 4 partes são usadas para treinar e 1 parte para validar.
# - O processo se repete 5 vezes, trocando a parte de validação a cada rodada.
# - O desempenho médio nessas 5 rodadas é usado para avaliar cada combinação de parâmetros.
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("GridSearchCV concluído.")

# Mostra os melhores parâmetros encontrados
print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)
