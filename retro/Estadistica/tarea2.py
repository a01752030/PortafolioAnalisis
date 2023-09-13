import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import probplot, norm
from scipy.stats import skew, kurtosis

M = pd.read_csv("mc-donalds-menu-1.csv")

X = M['Calories']
# 1. Cuartiles
q1 = X.quantile(0.25)
q3 = X.quantile(0.75)

y1 = X.min()
y2 = X.max()

# 2. Intercuartil
ri = q3 - q1

# 3. 2x1 grid
fig, axes = plt.subplots(2, 1)

# 4. boxplot
axes[0].boxplot(X, vert=False)
axes[0].set_xlim([y1, y2]) 

# 5. outliers
axes[0].axvline(x=q3 + 1.5 * ri, color='red')

# 6. Criterio de outlier
X1 = M[M['Calories'] < q3 + 1.5 * ri]['Calories']

# 7. Datos Filtrados
print(X1.describe())

# 8. Datos originales
print(X.describe())

# Plots
plt.tight_layout()
plt.show()


# Q-Q Plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
probplot(X, dist="norm", plot=plt)
plt.title("Q-Q Plot")

# Histogram with Normal Density Curve
plt.subplot(1, 2, 2)

# Histogram
plt.hist(X, bins=30, density=True, color='gray', alpha=0.7)

# Normal Density Curve
x = np.linspace(min(X), max(X), 100)
y = norm.pdf(x, np.mean(X), np.std(X))
plt.plot(x, y, 'r-', label="Normal Distribution")

plt.title("Histogram with Normal Density")
plt.legend()

plt.tight_layout()
plt.show()

# Python's skewness and kurtosis
skew_val = skew(X)
kurt_val = kurtosis(X, fisher=False) # fisher=False gives "regular" kurtosis; if True, it's excess kurtosis

print(f"Skewness: {skew_val}")
print(f"Kurtosis: {kurt_val}")