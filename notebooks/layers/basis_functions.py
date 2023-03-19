import numpy as np
from scipy.optimize import brentq
import tensorflow as tf

def fac(n):
    return 1 if n == 0 else n * fac(n - 1)

def j_l(l):
    # calculate coefficients of sin(x)/x**k, cos(x)/x**k terms for 1 <= k <= n + 1 
    # based on Rayleigh's formula
    array_length = 3 * l + 1
    coeffs = np.array([[c + r == 0 for c in range(2)] for r in range(array_length)], dtype=np.float32)
    for _ in range(l):
        for c in range(array_length):
            c = array_length - c - 1
            [lamb, alpha] = coeffs[c - 1]
            [x, y] = coeffs[c]
            coeffs[c-1] = [-alpha, lamb]
            coeffs[c] = [-c*lamb + x, -c*alpha + y]
        for c in range(array_length - 1):
            c = array_length - c - 1
            coeffs[c] = coeffs[c-1]
        coeffs[0] = [0.0, 0.0]
    for c in range(array_length - l):
        coeffs[c] = (-1) ** l * coeffs[c+l]

    def helper(x):
        return np.array([[np.sin(x) / x ** (i+1), np.cos(x) / x ** (i+1)] for i in range(array_length)])
    return np.vectorize(lambda x: np.sum(np.multiply(helper(x), coeffs))) 

def calculate_j_l_zeros(l, n):
    if l == 1:
        return np.array()
    zeros_init = np.array([[j * np.pi * (i == 0) for j in range(1, n+1+l)] for i in range(l+1)])
    for i in range(1, l+1):
        for j in range(l+n-i):
            a = zeros_init[i-1][j]
            b = zeros_init[i-1][j+1]
            zeros_init[i][j] = brentq(j_l(i), a, b)
    return zeros_init[-1,:n]

def legendre_polynomial_coeffs(l):
    # calculate coefficients of Legendre in monomial basis based on recurrence formula
    if l == 0:
        return [1.0]
    if l == 1:
        return [0, 1.0]
    coeffs_l_minus_one = legendre_polynomial_coeffs(l-1)
    coeffs_l_minus_one.insert(0, 0)
    coeffs_l_minus_two = legendre_polynomial_coeffs(l-2)
    coeffs_l_minus_two += [0.0, 0.0]
    res = [(2*l - 1) * coeffs_l_minus_one[i] - (l - 1) * coeffs_l_minus_two[i] for i in range(l + 1)]
    return [i/l for i in res]

def legendre_polynomial(l):
    coeffs = np.array(legendre_polynomial_coeffs(l))
    return lambda x: np.sum(np.multiply(np.array([x ** i for i in range(l+1)]), coeffs))

def Y_l_0(l):
    prefactor = np.sqrt((2*l + 1) / 4*np.pi)
    return np.vectorize(lambda x: prefactor * legendre_polynomial(l)(np.cos(x)))

def envelope_u(p):
    return np.vectorize(lambda d: (1 - (p+1)*(p+2)/2*d**p + p*(p+2)*d**(p+1) - p*(p+1)/2*d**(p+2)))

def a_sbf(l, n, d, alpha, c):
    z_l = calculate_j_l_zeros(l, n)
    z_ln = z_l[n-1]
    prefactor = np.sqrt(2 / (c**3 * j_l(l+1)(z_ln)**2)) 
    a_tilde_sbf = prefactor * j_l(l)(z_ln / c * d) * Y_l_0(l)(alpha)
    return a_tilde_sbf

def e_rbf(d, c):
    return np.sqrt(2/c) * tf.sin(d) / d