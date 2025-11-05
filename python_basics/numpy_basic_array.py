import numpy as np

# Criando um vetor de 100 elementos
y1 = np.arange(100)  # shape (100,)
y2 = np.arange(100).reshape(-1, 1)  # shape (100, 1)

print("Shape de y1:", y1.shape)
print("Primeira linha de y1:", y1[0])
print("Segunda linha de y1:", y1[1])
print("Ultima linha de y1:", y1[-1])
print("Shape de y2:", y2.shape)
print("Primeira linha de y2:", y2[0])
print("Segunda linha de y2:", y2[1])
print("Ultima linha de y2:", y2[-1])

# Exemplo de indexação
print("\nAcessando y1[0]:", y1[0])  # retorna um escalar
print("Acessando y2[0]:", y2[0])  # retorna um array de 1 elemento
print("Acessando y2[0, 0]:", y2[0, 0])  # retorna um escalar

# Exemplo de uso em operações
print("\nSoma de todos elementos y1:", y1.sum())
print("Soma de todos elementos y2:", y2.sum())
