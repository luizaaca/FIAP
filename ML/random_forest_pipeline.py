from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Carrega o dataset Iris (flores), já separado em X (características) e y (rótulos)
X, y = load_iris(return_X_y=True)
print("Formato de X:", X.shape, ", Formato de y:", y.shape)

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

# Treina o pipeline com os dados de treino
print("Treinando o pipeline...")
pipeline.fit(X_train, y_train)
print("Pipeline treinado.")

# Faz previsões com os dados de teste
print("Realizando previsões nos dados de teste...")
predictions = pipeline.predict(X_test)
print("Previsões concluídas.")

# Mostra as previsões
print("Previsões do conjunto de teste:")
print(predictions)
print("Valores reais do conjunto de teste:")
print(y_test)
