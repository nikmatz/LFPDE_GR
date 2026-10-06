# =====================================================================
#  Κεφάλαιο 6 — Φάσματα, Parseval και FFT με SymPy, NumPy και SciPy
#  Συνοδευτικός κώδικας του βιβλίου «Μετασχηματισμοί Laplace και Fourier,
#  Μερικές Διαφορικές Εξισώσεις», Ν. Ματζάκος.
#  Τρέχει όπως είναι σε Python 3 (sympy, numpy, scipy, matplotlib)
#  ή απευθείας στη σελίδα του βιβλίου (Pyodide).
# =====================================================================
import sympy as sp
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

t = sp.symbols('t', real=True)
n = sp.symbols('n', integer=True, nonzero=True)

# ---------- Α. Ο FFT στην πράξη (Ενότητα 6.6) ----------
for N in (64, 256, 1024):
    tm = 2*np.pi*np.arange(N)/N                 # t_m = mT/N, T = 2π
    xs = np.where(tm < np.pi, 1.0, -1.0)
    xs[0] = xs[N//2] = 0.0                      # μέση τιμή στα άλματα
    A = 2*np.abs(np.fft.fft(xs)[1:10])/N        # A_k ≈ 2|X_k|/N
    print(f"Α. N = {N:4d}:", np.round(A[[0, 2, 4, 6, 8]], 4))
print("   ακριβή 4/(πk):", np.round(4/(np.pi*np.arange(1, 10, 2)), 4))

# ---------- Β. Λυμένη Άσκηση 6.1: c_n, πλάτη και φάσεις ----------
cn = sp.simplify(sp.integrate(t*sp.exp(-sp.I*n*sp.pi*t), (t, 0, 1))/2)
print("\nΒ. c_n =", cn)
k = np.arange(1, 8)
c = np.array([complex(cn.subs(n, int(m))) for m in k])
An, phin = 2*np.abs(c), -np.angle(c)
print("   A_n =", np.round(An, 3))
print("   φ_n =", np.round(phin, 3))
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
ax[0].stem(np.r_[0, k], np.r_[0.25, An]); ax[0].set_title("A_n"); ax[0].set_xlabel("n")
ax[1].stem(k, phin); ax[1].set_title("φ_n"); ax[1].set_xlabel("n")
plt.tight_layout(); plt.show()

# ---------- Γ. Ταυτότητα Parseval για την ίδια συνάρτηση ----------
P = quad(lambda s: s*s, 0, 1)[0]/2              # μέση ισχύς: 1/6
for M in (10, 100, 1000):
    m = np.arange(1, M + 1)
    cm = 1j*(-1.0)**m/(2*m*np.pi) - (1 - (-1.0)**m)/(2*m**2*np.pi**2)
    print(f"Γ. M = {M:4d}:  P - Σ|c_n|² = {P - (0.25**2 + 2*np.sum(np.abs(cm)**2)):.2e}")

# ---------- Δ. Παράδειγμα 6.6: παλμοί ύψους 2, διάρκειας 1, περιόδου 3 ----------
k = np.arange(1, 9)
c = 2/(np.pi*k)*np.sin(k*np.pi/3)*np.exp(-1j*k*np.pi/3)
A = 2*np.abs(c); ph = np.where(A > 1e-12, -np.angle(c), np.nan)
print("\nΔ. A_n =", np.round(A, 4))
print("   φ_n/π =", np.round(ph/np.pi, 4))

# ---------- Ε. Παράδειγμα 6.4: από το φάσμα στο σήμα ----------
c1, c3 = 1 - 1j, -np.sqrt(3)/2 + 0.5j
tt = np.linspace(0, 2*np.pi, 7)
f_exp = -0.5 + 2*np.real(c1*np.exp(1j*tt) + c3*np.exp(3j*tt))
f_amp = -0.5 + 2*np.sqrt(2)*np.cos(tt - np.pi/4) + 2*np.cos(3*tt + 5*np.pi/6)
print("\nΕ. μέγιστη διαφορά των δύο μορφών:", np.max(np.abs(f_exp - f_amp)))
print("   atan2(b3, a3) =", np.arctan2(-1, -np.sqrt(3)), " = -5π/6 =", -5*np.pi/6)
