## Naive Bayes — Explicação didática

Naive Bayes é uma família de algoritmos de classificação probabilísticos baseados no Teorema de Bayes, com a suposição ("naive") de independência condicional entre features. Apesar dessa suposição forte, Naive Bayes costuma funcionar bem em diversas aplicações práticas, especialmente em texto e classificação de alto dimensionalidade.

- Intuição e objetivo:
	- O classificador estima a probabilidade posterior de cada classe dado um vetor de features e escolhe a classe com maior probabilidade.
	- Usa P(C|X) ∝ P(X|C) P(C) (Teorema de Bayes). As estimativas de P(X|C) são simplificadas pela suposição de independência entre features para computação eficiente.

- Teorema de Bayes (fórmula):
	\(\;P(C\mid X)=\frac{P(X\mid C)\,P(C)}{P(X)}\;\)
	- Para classificação, comparamos as quantidades \(P(X\mid C)P(C)\) entre classes (o denominador P(X) é comum e pode ser ignorado ao escolher o máximo).

- Suposição "naive":
	- Assume-se que as features são condicionalmente independentes dado a classe: \(P(X\mid C)=\prod_{i} P(x_i\mid C)\).
	- Essa hipótese facilita cálculo e modelagem, embora raramente seja estritamente verdadeira.

- Variantes populares:
	- Gaussian Naive Bayes:
		- Assume que cada feature contínua segue uma distribuição normal (gaussiana) condicionada à classe.
		- Estima média e variância por feature e classe; avalia \(P(x_i\mid C)\) via densidade normal.
	- Multinomial Naive Bayes:
		- Usado frequentemente em dados de contagem (ex.: bag-of-words em NLP).
		- Modela a probabilidade de palavras (ou tokens) condicionadas à classe.
	- Bernoulli Naive Bayes:
		- Para features binárias (presença/ausência).
		- Modela cada feature como distribuição de Bernoulli condicionada à classe.

- Estimativa de probabilidades e suavização:
	- Contagens de ocorrência podem levar a probabilidades zero (problema em Multinomial/Bernoulli).
	- Suavização Laplace (add-one) ou Lidstone (add-α) é usada para evitar zeros:
		\(\hat{P}(w\mid C)=\frac{N_{w,C}+\alpha}{N_C + \alpha V}\)
		- onde N_{w,C} é contagem da palavra w em documentos da classe C, N_C soma das contagens na classe, V é tamanho do vocabulário, e α é o parâmetro de suavização (α=1 → Laplace).

- Vantagens:
	- Simples, rápido e eficiente em memória e tempo (treino e predição rápidos).
	- Funciona bem em alta dimensionalidade (ex.: texto), quando as features são (aproximadamente) independentes.
	- Pouco ou nenhum ajuste de hiperparâmetros em muitos casos.

- Desvantagens e limitações:
	- Suposição de independência frequentemente violada; apesar disso, o classificador pode continuar performando bem.
	- Para relações complexas entre features, modelos discriminativos (SVM, redes neurais) podem superar o desempenho.
	- Estimativas de probabilidade podem ser ruins (mal calibradas), embora a ordenação das classes normalmente seja adequada.

- Métricas de avaliação:
	- Acurácia, precisão, recall, F1-score (para classificação binária/multiclasse).
	- Curva ROC / AUC para problemas binários.

- Quando usar Naive Bayes:
	- Baseline rápido para tarefas de classificação (ex.: categorização de texto, spam detection).
	- Cenários com muitas features e poucos dados por instância.
	- Sistemas onde velocidade e simplicidade são prioritárias.

- Dicas práticas:
	- Para texto, experimente MultinomialNB com TF or TF-IDF e suavização α via validação cruzada.
	- Para features contínuas que não parecem gaussianas, considere transformar variáveis (log, box-cox) ou usar discretização.
	- Verifique o impacto da suavização α; valores pequenos (ex.: 1e-3 a 1) podem ajudar em vocabulários grandes.
	- Balanceamento de classes: Se houver desbalanceamento forte, considere ajustar prioridades (class_prior) ou usar técnicas de reamostragem.

- Exemplo rápido (Scikit-learn — Multinomial Naive Bayes em texto):

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# X_text: lista de documentos (strings), y: rótulos
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=0)

vectorizer = TfidfVectorizer(max_features=20000)
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

clf = MultinomialNB(alpha=1.0)
clf.fit(X_train_tf, y_train)

preds = clf.predict(X_test_tf)
print(classification_report(y_test, preds))
```

- Observações finais:
	- Apesar da simplicidade e da suposição forte, Naive Bayes continua sendo uma ferramenta prática e robusta em muitos cenários. Use-o como baseline e para aplicações que exigem modelos rápidos e interpretáveis.

## Exemplo didático (texto)

Problema: classificar mensagens como "Spam" (S) ou "Ham" (H) usando Multinomial Naive Bayes com suavização Laplace (α = 1).

Conjunto de treino (contagens de palavras relevantes: free, win, meeting, project):

- Doc1 (Spam): "free win" → contagens: free=1, win=1, meeting=0, project=0
- Doc2 (Spam): "free" → contagens: free=1, win=0, meeting=0, project=0
- Doc3 (Ham): "meeting project" → contagens: free=0, win=0, meeting=1, project=1
- Doc4 (Ham): "project" → contagens: free=0, win=0, meeting=0, project=1
- Doc5 (Ham): "meeting" → contagens: free=0, win=0, meeting=1, project=0

Passo 1 — Priori das classes:
- P(S) = 2/5 (2 documentos Spam de 5)
- P(H) = 3/5

Passo 2 — somas de contagens por classe (N_C) e vocabulário V = 4 palavras:
- Para Spam: contagem total de palavras N_S = 1+1 + 1 = 3 (Doc1 e Doc2)
- Para Ham: N_H = 1+1 + 1 = 3 (Doc3, Doc4, Doc5)

Com suavização Laplace (α = 1), probabilidade de uma palavra w dado a classe C em MultinomialNB é
P(w|C) = (N_{w,C} + α) / (N_C + α V).

Calculemos probabilidades essenciais:
- Para Spam:
	- P(free|S) = (2 + 1) / (3 + 1*4) = 3/7
	- P(win|S) = (1 + 1) / (3 + 4) = 2/7
	- P(meeting|S) = (0 + 1) / 7 = 1/7
	- P(project|S) = (0 + 1) / 7 = 1/7
- Para Ham:
	- P(free|H) = (0 + 1) / (3 + 4) = 1/7
	- P(win|H) = (0 + 1) / 7 = 1/7
	- P(meeting|H) = (2 + 1) / 7 = 3/7
	- P(project|H) = (2 + 1) / 7 = 3/7

Passo 3 — classificar nova mensagem M = "free win":

Calcular pontuação proporcional à posterior: score(C) = P(C) * Π P(word|C).

- score(Spam) ∝ P(S) * P(free|S) * P(win|S) = (2/5) * (3/7) * (2/7) = (2/5) * 6/49 = 12/245 ≈ 0.0490
- score(Ham)  ∝ P(H) * P(free|H) * P(win|H) = (3/5) * (1/7) * (1/7) = (3/5) * 1/49 = 3/245 ≈ 0.0122

Comparando as pontuações, score(Spam) > score(Ham), logo a predição é "Spam".

Observação: ignoramos o denominador P(M) comum às classes; a ordem relativa das pontuações determina a classe predita. A suavização evita probabilidades zero e permite lidar com palavras não vistas em uma classe.

