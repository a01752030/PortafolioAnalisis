import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm,t,chi2

'''
/*----------------------------------------------------------
 * Tarea #1: Algunas distribuciones de probabilidad
 *
 * Fecha: 16-Aug-2023
 * Authors:
 *           A01752030 Juan Pablo Castañeda Serrano
 *----------------------------------------------------------*/
'''

def ejercicio1():  
    # Parámetros
    mu = 10  # Media
    sigma = 2  # Desviación estándar

    # Genera valores para el eje x
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)

    # Calcula la PDF (Probability Density Function) de la distribución normal para cada valor x
    y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(- 0.5 * ((x - mu)/sigma)**2)

    # Graficar
    plt.plot(x, y, lw=2)
    plt.title('Distribución Normal')
    plt.xlabel('x')
    plt.ylabel('Densidad de probabilidad')
    plt.grid(True)
    plt.show()

def ejercicio2():
    # Grados de libertad
    v = 12

    # Gamma function using numpy
    gamma = np.math.gamma

    # PDF of t-distribution
    def t_pdf(t, v):
        numerator = gamma((v + 1) / 2)
        denominator = np.sqrt(v * np.pi) * gamma(v / 2)
        return numerator / denominator * (1 + t**2 / v) ** (-0.5 * (v + 1))

# Generate values for x-axis
    x = np.linspace(-5, 5, 1000)

# Calculate PDF values
    y = [t_pdf(i, v) for i in x]

# Plot
    plt.plot(x, y, lw=2)
    plt.title("Student's t-Distribution with v=12 degrees of freedom")
    plt.xlabel('t')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.show()    

def ejercicio3():
    k = 8
    gamma = np.math.gamma

    def chi2_pdf(x, k):
        return (1 / (2**(k/2) * gamma(k/2))) * x**(k/2 - 1) * np.exp(-x/2)
    x = np.linspace(0, 20, 1000)

# Calcular valores de la PDF
    y = [chi2_pdf(i, k) for i in x]

# Graficar
    plt.plot(x, y, lw=2)
    plt.title('Distribución Chi-cuadrada con k=8 grados de libertad')
    plt.xlabel('x')
    plt.ylabel('Densidad de probabilidad')
    plt.grid(True)
    plt.show()


def ejercicio4():
    z_value = norm.ppf(0.45)
    print(z_value)


def ejercicio5():
    mu = 100
    sigma = 7
    prob1 = norm.cdf(87, mu, sigma)
    prob2 = 1 - prob1
    prob3 = norm.cdf(110, mu, sigma) - prob1
    print(f'P(X < 87) = {prob1:.6f}')
    print(f'P(X > 87) = {prob2:.6f}')
    print(f'P(87 < X < 110) = {prob3:.6f}')

def ejercicio6():
    # Grados de libertad
    gl = 10

# P(X < 0.5)
    prob1 = t.cdf(0.5, gl)

# P(X > 1.5)
    prob2 = 1 - t.cdf(1.5, gl)

# Valor t para el cual P(X < t) = 0.05
    t_value = t.ppf(0.05, gl)

    print(f'P(X < 0.5) = {prob1:.7f}')
    print(f'P(X > 1.5) = {prob2:.7f}')
    print(f't for P(X < t) = 0.05: {t_value:.6f}')


def ejercicio7():
    gl = 6

# P(X^2 < 3)
    prob1 = chi2.cdf(3, gl)

# P(X^2 > 2)
    prob2 = 1 - chi2.cdf(2, gl)

    print(f'P(X^2 < 3) = {prob1:.7f}')
    print(f'P(X^2 > 2) = {prob2:.7f}')



if __name__ == '__main__':
    ejercicio1()
    ejercicio2()
    ejercicio3()
    #template()
