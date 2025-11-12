import pandas as pd

dados = pd.Series([5, 12, 18, 23, 37])
faixas_nomeadas = pd.cut(dados, bins=[0, 10, 20, 40], labels=['baixo', 'médio', 'alto'])
print("Valores originais:")
print(dados.values)
print("\nFaixas categóricas:")
print(faixas_nomeadas)
