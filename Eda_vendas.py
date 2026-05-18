import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

df = pd.read_csv("vendas_rede_lojas.csv")
print("Dataset carregado com sucesso")

df.head(10)

df.info()

# -------------------------------------------------------
# LIMPEZA DE DADOS
# -------------------------------------------------------

# Valores ausentes — uso da mediana por ser mais robusta a outliers
df["Desconto_%"] = df["Desconto_%"].fillna(df["Desconto_%"].median())
df["Quantidade"] = df["Quantidade"].fillna(df["Quantidade"].median())
df["Preco_Unitario"] = df["Preco_Unitario"].fillna(df["Preco_Unitario"].median())

# Coluna Vendedor removida pois é texto e não contribui para a análise numérica
df = df.drop("Vendedor", axis=1)

# Convertendo a coluna Data para o formato datetime
df["Data"] = pd.to_datetime(df["Data"], format="%Y-%m-%d")

df.info()

# -------------------------------------------------------
# CRIAÇÃO DE COLUNAS
# -------------------------------------------------------

# Valor total já considerando o desconto aplicado
df["Valor_Total"] = (df["Quantidade"] * df["Preco_Unitario"]) * (1 - df["Desconto_%"] / 100)

# Valor bruto sem desconto, para comparar o impacto dos descontos
df["Valor_Bruto"] = df["Quantidade"] * df["Preco_Unitario"]

df.info()

# -------------------------------------------------------
# ESTATÍSTICAS DESCRITIVAS — VALOR TOTAL
# -------------------------------------------------------

print("Média:", df["Valor_Total"].mean().round(2))
print("Mediana:", df["Valor_Total"].median().round(2))
print("Desvio padrão:", df["Valor_Total"].std().round(2))
print("Valor máximo:", df["Valor_Total"].max().round(2))
print("Valor mínimo:", df["Valor_Total"].min().round(2))

df.describe()

# -------------------------------------------------------
# ANÁLISE DE OUTLIERS — MÉTODO IQR
# -------------------------------------------------------

# Outliers em Valor Total
q1 = df["Valor_Total"].quantile(0.25)
q2 = df["Valor_Total"].quantile(0.50)
q3 = df["Valor_Total"].quantile(0.75)
iqr = q3 - q1

limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr  # corrigido: antes estava sem o q3

print(f"Q1: {q1:.2f}")
print(f"Q2 (Mediana): {q2:.2f}")
print(f"Q3: {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Limite inferior: {limite_inferior:.2f}")
print(f"Limite superior: {limite_superior:.2f}")

outliers_valor = df[(df["Valor_Total"] < limite_inferior) | (df["Valor_Total"] > limite_superior)]
print(f"Outliers encontrados em Valor Total: {len(outliers_valor)}")

# Outliers em Quantidade
q1 = df["Quantidade"].quantile(0.25)
q2 = df["Quantidade"].quantile(0.50)
q3 = df["Quantidade"].quantile(0.75)
iqr = q3 - q1

limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

print(f"\nQ1: {q1:.2f}")
print(f"Q2 (Mediana): {q2:.2f}")
print(f"Q3: {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Limite inferior: {limite_inferior:.2f}")
print(f"Limite superior: {limite_superior:.2f}")

outliers_qtd = df[(df["Quantidade"] < limite_inferior) | (df["Quantidade"] > limite_superior)]
print(f"Outliers encontrados em Quantidade: {len(outliers_qtd)}")

# -------------------------------------------------------
# ANÁLISE POR CATEGORIA
# -------------------------------------------------------

print("\nMediana de Quantidade por Categoria:")
print(df.groupby("Categoria")["Quantidade"].median())
# Acessórios tem a maior mediana

print("\nValor Total máximo por Categoria:")
print(df.groupby("Categoria")["Valor_Total"].max())

print("\nValor Total médio por Categoria:")
print(df.groupby("Categoria")["Valor_Total"].mean().round(2))
# Eletrônicos tem o maior ticket médio

# -------------------------------------------------------
# VISUALIZAÇÕES
# -------------------------------------------------------

# Distribuição de Valor Total por Região
plt.figure(figsize=(10, 5))
sns.boxplot(x="Regiao", y="Valor_Total", data=df)
plt.title("Valor Total por Região")
plt.xlabel("Região")
plt.ylabel("Valor Total (R$)")
plt.tight_layout()
plt.show()
# Região Central apresenta maior variabilidade e mediana mais alta

# Quantidade vs Valor Total por Categoria
plt.figure(figsize=(10, 5))
sns.scatterplot(x="Valor_Total", y="Quantidade", data=df, hue="Categoria")
plt.title("Quantidade x Valor Total por Categoria")
plt.xlabel("Valor Total (R$)")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()
# Eletrônicos concentra as transações com valores mais altos

# Desconto vs Valor Total
plt.figure(figsize=(10, 5))
sns.scatterplot(x="Desconto_%", y="Valor_Total", data=df, hue="Categoria")
plt.title("Desconto (%) x Valor Total")
plt.xlabel("Desconto (%)")
plt.ylabel("Valor Total (R$)")
plt.tight_layout()
plt.show()
# Não há correlação clara entre desconto e valor total

# -------------------------------------------------------
# CORRELAÇÕES
# -------------------------------------------------------

print("\nMatriz de correlação:")
correlation_matrix = df[["Quantidade", "Preco_Unitario", "Desconto_%", "Valor_Total"]].corr()
print(correlation_matrix.round(2))

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# ANÁLISE TEMPORAL
# -------------------------------------------------------

# Extraindo o mês e ano para agrupar as vendas no tempo
df["AnoMes"] = df["Data"].dt.to_period("M")

vendas_por_mes = df.groupby("AnoMes")["Valor_Total"].sum()

plt.figure(figsize=(12, 5))
vendas_por_mes.plot(marker="o", color="#2d7dd2")
plt.title("Evolução Mensal do Faturamento")
plt.ylabel("Valor Total (R$)")
plt.xlabel("Mês")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# CONCLUSÕES
# -------------------------------------------------------

# A: Coluna Desconto_% tinha 165 dados faltantes — imputados com mediana
# B: Foram encontrados outliers em Valor_Total e Quantidade
# C: Não há correlação significativa entre desconto e valor total
#    o que pode indicar que os descontos não estão sendo bem direcionados
# D: Região Central teve o maior desempenho em variabilidade e mediana de vendas
# E: Eletrônicos teve o maior ticket médio entre as categorias
# F: A análise temporal permite identificar sazonalidade nas vendas —
#    útil para planejar estoque e campanhas promocionais
