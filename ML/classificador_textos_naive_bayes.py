from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np

# Dados de exemplo
textos = [
    "O novo lançamento da Apple",
    "Resultado do jogo de ontem",
    "Eleições presidenciais",
    "Atualização no mundo da tecnologia",
    "Campeonato de futebol",
    "Política internacional",
    # adicionando mais exemplos, com palavras diferentes, faz a acurácia cair drasticamente
    "Ciencia e inovação tecnológica",
    "Basquete: melhores momentos",
    "Debates políticos acalorados",
]
categorias = [
    "tecnologia",
    "esportes",
    "política",
    "tecnologia",
    "esportes",
    "política",
    "tecnologia",
    "esportes",
    "política",
]

# Convertendo textos em uma matriz de contagens de tokens
# O CountVectorizer transforma cada frase em um vetor de números.
# Cada posição do vetor representa uma palavra única encontrada em todos os textos.
# O valor 1 indica que a palavra aparece na frase; 0 indica que não aparece.
# Assim, cada frase vira um array de números que o computador pode usar para aprender.
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)
print(vectorizer.get_feature_names_out())
print("Matriz de características:\n", X.toarray(), "\n")  # type: ignore
# Dividindo os dados em conjuntos de treinamento e teste
# O parâmetro random_state define uma "semente" para o gerador de números aleatórios,
# garantindo que a divisão dos dados seja sempre igual toda vez que o código for executado.
X_train, X_test, y_train, y_test = train_test_split(
    X, categorias, test_size=0.5, random_state=42
)
# print("Conjunto de treinamento:\n", X_train.toarray(), "\n")
# print("Conjunto de teste:\n", X_test.toarray(), "\n")
# print("Rótulos de treinamento:\n", y_train, "\n")
# print("Rótulos de teste:\n", y_test, "\n")

# Treinando o classificador
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predição e Avaliação
y_pred = clf.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")

# O Naive Bayes aprende, durante o treinamento, quantas vezes cada palavra aparece em cada categoria.
# Quanto mais uma palavra aparece em frases de uma categoria, maior será a chance (probabilidade)
# de uma nova frase com essa palavra ser classificada nessa mesma categoria.
# Por exemplo: se "futebol" aparece muito em "esportes", frases com "futebol" terão mais chance de serem "esportes".
# Assim, o modelo não faz uma busca exata por palavras, mas sim uma análise estatística simples das palavras.

# Para visualizar as probabilidades aprendidas para cada palavra em cada categoria:
feature_names = vectorizer.get_feature_names_out()
classes = clf.classes_
# clf.feature_log_prob_ contém o log das probabilidades de cada palavra em cada categoria
print("\nProbabilidades das palavras em cada categoria:")
for i, classe in enumerate(classes):
    print(f"\nCategoria: {classe}")
    for j, palavra in enumerate(feature_names):
        prob = np.exp(clf.feature_log_prob_[i, j])
        print(f"  Palavra '{palavra}': {prob:.4f}")
