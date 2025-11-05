## Regressão Linear — Explicação didática

Regressão linear é um método estatístico e de machine learning usado para modelar a relação entre uma variável dependente (alvo) contínua e uma ou mais variáveis independentes (features). A ideia central é ajustar uma função linear que prevê o valor esperado da saída a partir das entradas.

- Estrutura básica:
	- Modelo simples (univariado): y = b0 + b1 * x
	- Modelo multivariado (múltiplas features): y = b0 + b1*x1 + b2*x2 + ... + bn*xn
	- b0 é o intercepto (bias) e b1..bn são os coeficientes (pesos) aprendidos.

- Ideia intuitiva:
	1. Procuramos uma linha (ou hiperplano) que passe pelos dados de forma que as previsões fiquem o mais próximas possível dos valores reais.
	2. Medimos o erro entre previsões e valores reais (por exemplo, com erro quadrático médio).
	3. Ajustamos os coeficientes para minimizar esse erro — isto é a estimação do modelo.

- Função de perda comum:
	- Mean Squared Error (MSE):
		\(\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2\)
	- O objetivo do Ordinary Least Squares (OLS) é minimizar a soma dos quadrados dos resíduos.

- Ajuste do modelo (estimar coeficientes):
	- Solução analítica (OLS):
		- Em notação matricial, com X a matriz de features (com coluna de 1s para o intercepto) e y o vetor de alvos, a solução é
			\(\beta = (X^T X)^{-1} X^T y\)
		- Rápida e exata quando (X^T X) é invertível e o número de features não é muito grande.
	- Métodos iterativos (gradiente):
		- Gradient Descent: útil quando há muitas features ou muitos dados. Atualiza os coeficientes na direção do gradiente da perda.

- Regularização (quando evitar overfitting / multicolinearidade):
	- Ridge (L2): adiciona \(\lambda \sum_j \beta_j^2\) à função de perda — penaliza pesos grandes.
	- Lasso (L1): adiciona \(\lambda \sum_j |\beta_j|\) — pode zerar coeficientes, realizando seleção de features.
	- Elastic Net: combinação L1 + L2.

- Suposições clássicas da regressão linear (para inferência e interpretação):
	1. Linearidade: relação linear entre features e target.
	2. Independência: observações independentes.
	3. Homocedasticidade: variância constante dos resíduos.
	4. Normalidade dos erros: resíduos normalmente distribuídos (importante para testes estatísticos e intervalos de confiança).
	5. Ausência de multicolinearidade severa entre features.

- Métricas de avaliação:
	- Mean Squared Error (MSE)
	- Root Mean Squared Error (RMSE)
	- Mean Absolute Error (MAE)
	- R-squared (R²): proporção da variância explicada pelo modelo.

- Vantagens:
	- Simples de entender e interpretar — coeficientes têm interpretação direta.
	- Rápida de treinar (solução fechada quando aplicável).
	- Base para modelos mais complexos e técnicas de inferência estatística.

- Desvantagens e limitações:
	- Assume linearidade — relações não-lineares não são capturadas.
	- Sensível a outliers (MSE amplifica grandes erros).
	- Se as suposições clássicas não se mantêm, inferência e intervalos podem ser inválidos.

- Quando usar:
	- Bom ponto de partida para problemas de regressão.
	- Útil quando interpretabilidade é importante.
	- Use como baseline antes de modelos mais complexos (árvores, redes neurais, ensembles).

- Exemplo rápido (Scikit-learn):

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# X: features (n_samples, n_features), y: valores contínuos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print('RMSE:', mean_squared_error(y_test, preds, squared=False))
print('R2:', r2_score(y_test, preds))
```

- Dicas práticas:
	- Sempre visualize a relação entre cada feature e o target (scatterplots) para checar linearidade.
	- Escale features quando usar regularização (Ridge/Lasso) para que a penalização afecte igualmente as variáveis.
	- Verifique resíduos (residual plots) para checar homocedasticidade e padrões não capturados pelo modelo.
	- Trate outliers apropriadamente (transformações, winsorizing ou modelos robustos).
	- Use validação cruzada ao ajustar hiperparâmetros de regularização (por exemplo, alpha em Ridge/Lasso).

## Exemplo didático (texto)

Problema simples: ajustar uma reta (regressão linear univariada) usando dois pontos e usar a reta ajustada para prever um novo valor.

Dados observados (x, y):

- P1: (x = 1, y = 3)
- P2: (x = 3, y = 7)

Passo 1 — encontrar a inclinação (coeficiente b1):
- A inclinação b1 é a variação de y sobre a variação de x entre os dois pontos: b1 = (7 - 3) / (3 - 1) = 4 / 2 = 2.

Passo 2 — calcular o intercepto (b0):
- Usando P1: 3 = b0 + 2*1 ⇒ b0 = 1.

Logo a equação ajustada é: y = 1 + 2x.

Previsão: para um novo valor x = 2, estimamos y = 1 + 2*2 = 5.

Interpretação: o coeficiente b1 = 2 indica que, em média, cada aumento de 1 unidade em x está associado a um aumento de 2 unidades em y. Com apenas dois pontos a reta passa exatamente por ambos (ajuste perfeito); em dados reais com ruído usa-se a minimização dos resíduos quadráticos para encontrar os coeficientes que melhor explicam os dados.

Este exemplo ilustra o conceito básico de ajuste linear e como usar a reta estimada para prever novos valores, sem entrar em detalhes algébricos ou computacionais.


