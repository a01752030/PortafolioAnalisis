import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Cargar el dataset
df = pd.read_csv('precios_autos.csv')

# Observar dimensiones del dataset
print(df.shape)

# Observar el tipo de dato que se tiene para cada una de las columnas en nuestro dataset
print(df.dtypes)

# Convirtiendo variable de texto categóricas a numéricas categóricas
cylinder_mapping = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "twelve": 12}
df['cylindernumber'] = df['cylindernumber'].map(cylinder_mapping)

# Buscar valores nulos dentro del dataset
print(df.isnull().sum())

# Filtrar columnas numéricas
numeric_cols = df.select_dtypes(include=[np.number])

# Gráficos de histogramas para variables numéricas
for col in numeric_cols.columns:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()

# Medidas estadísticas apropiadas para las variables cuantitativas
quantitative_vars = ["wheelbase", "carlength", "carwidth", "carheight", "curbweight", "enginesize", "stroke",
                     "compressionratio", "horsepower", "peakrpm", "citympg", "highwaympg", "price"]

for col in quantitative_vars:
    print(f"Variable: {col}")
    print(df[col].describe())
    print("\n")

# Medidas estadísticas apropiadas para las variables cualitativas
qualitative_vars = ["symboling", "fueltype", "carbody", "drivewheel", "enginelocation", "enginetype"]

for col in qualitative_vars:
    print(f"Frecuencias de {col}:\n{df[col].value_counts()}\n")
    print(f"Proporciones de {col}:\n{df[col].value_counts(normalize=True)}\n")
    print(f"Cuantiles de {col}:\n{df[col].quantile([0.25, 0.5, 0.75])}\n")

# Boxplots para variables cuantitativas
for col in quantitative_vars:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Matriz de correlación
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación")
plt.show()

# ANOVA para estimar el cambio de 'price' de acuerdo a la variable categórica 'symboling'
model = ols('price ~ C(symboling)', data=df).fit()
anova_table = anova_lm(model)
print(anova_table)

# Análisis de regresión lineal para variables numéricas transformadas
X = df[['curbweight', 'highwaympg', 'horsepower', 'wheelbase', 'carheight', 'enginesize']]
y = df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linear_model = LinearRegression()
linear_model.fit(X_scaled, y)

print("Coeficientes del modelo:")
print(linear_model.coef_)
print("Intercepto del modelo:", linear_model.intercept_)
print("R-squared:", linear_model.score(X_scaled, y))

# Gráfico de regresión lineal para una variable (por ejemplo, curbweight)
plt.figure()
sns.scatterplot(x=df['curbweight'], y=df['price'])
sns.regplot(x=df['curbweight'], y=df['price'], scatter=False, color='red')
plt.xlabel('Curb Weight')
plt.ylabel('Price')
plt.title('Regresión Lineal de Curb Weight vs Price')
plt.show()