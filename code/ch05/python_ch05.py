# =====================================================================
#  Κεφάλαιο 5 — Σειρές Fourier με SymPy, φαινόμενο Gibbs με NumPy
#  Συνοδευτικός κώδικας του βιβλίου «Μετασχηματισμοί Laplace και Fourier,
#  Μερικές Διαφορικές Εξισώσεις», Ν. Ματζάκος.
#  Τρέχει όπως είναι σε Python 3 (sympy, numpy, scipy, matplotlib)
#  ή απευθείας στη σελίδα του βιβλίου (Pyodide).
# =====================================================================
import sympy as sp
import numpy as np
from scipy.special import sici
import matplotlib.pyplot as plt

x = sp.symbols('x', real=True)
n = sp.symbols('n', integer=True, positive=True)
pi = sp.pi

# ---------- Α. Παράδειγμα 5.8: η σειρά της x^2 + x ----------
s1 = sp.fourier_series(x**2 + x, (x, -pi, pi))
print("Α. x^2 + x  ~ ", s1.truncate(5), " + ...")
an = sp.simplify(sp.integrate((x**2 + x)*sp.cos(n*x), (x, -pi, pi))/pi)
bn = sp.simplify(sp.integrate((x**2 + x)*sp.sin(n*x), (x, -pi, pi))/pi)
print("   a_n =", an, ",   b_n =", bn)

# ---------- Β. Παράδειγμα 5.9: σήμα δύο σταθμών ----------
s2 = sp.fourier_series(sp.Piecewise((-1, x < 0), (2, True)), (x, -pi, pi))
print("\nΒ. δύο στάθμες ~ ", s2.truncate(4), " + ...")

# ---------- Γ. Παράδειγμα 5.5: ολοκληρώματα με ορθογωνιότητα ----------
f = -1 + 3*sp.cos(x) + 2*sp.sin(2*x)
g = (2*sp.cos(x) - sp.sin(x))**2
print("\nΓ. g =", sp.simplify(sp.expand(sp.fu(g))))
print("   ∫f² =", sp.integrate(f**2, (x, -pi, pi)), ",   ∫fg =", sp.integrate(f*g, (x, -pi, pi)))

# ---------- Δ. Παράδειγμα 5.14: ημιτονικό ανάπτυγμα της cos x ----------
b = [sp.integrate(2/pi*sp.cos(x)*sp.sin(k*x), (x, 0, pi)) for k in range(1, 9)]
print("\nΔ. b_1..b_8 =", b)
K = np.arange(1, 20001)
print("   άθροισμα στο x=π/4: %.5f   (cos(π/4) = %.5f)" % (8/np.pi*np.sum(K*np.sin(K*np.pi/2)/(4*K**2 - 1)), np.cos(np.pi/4)))

# ---------- Ε. Πρόταση 5.3: το φαινόμενο Gibbs ----------
xx = np.linspace(1e-6, np.pi/2, 20001)
print("\nΕ.   N    max S_N     θέση      π/(N+1)")
for N in (9, 49, 199):
    k = np.arange(1, (N + 1)//2 + 1)
    S = 4/np.pi*np.sum(np.sin(np.outer(2*k - 1, xx))/(2*k - 1)[:, None], axis=0)
    print(f"   {N:3d}   {S.max():.4f}   {xx[S.argmax()]:.5f}   {np.pi/(N + 1):.5f}")
print("   όριο (2/π)Si(π) = %.5f" % (2/np.pi*sici(np.pi)[0]))

# ---------- ΣΤ. Παράδειγμα 5.12: η πριονωτή κοντά στο π ----------
for N in (10, 50, 200):
    m = np.arange(1, N + 1)
    xN = np.pi - np.pi/(N + 1)
    print(f"ΣΤ. N = {N:3d}:  S_N(x_N) = {2*np.sum((-1.0)**(m + 1)*np.sin(m*xN)/m):.4f}")
print("    όριο 2Si(π) = %.4f" % (2*sici(np.pi)[0]))

# ---------- Ζ. Σχήμα 5.5: μερικές σειρές του τετραγωνικού παλμού ----------
X = np.linspace(-np.pi, np.pi, 1201)
plt.figure(figsize=(7, 3.6))
plt.plot(X, np.sign(X), "k", lw=2, alpha=.35, label="q(x)")
for N in (1, 3, 7, 49):
    k = np.arange(1, (N + 1)//2 + 1)
    plt.plot(X, 4/np.pi*np.sum(np.sin(np.outer(2*k - 1, X))/(2*k - 1)[:, None], axis=0), label=f"S_{N}")
plt.axhline(1.179, ls="--", c="r", lw=.8)
plt.xlabel("x"); plt.grid(alpha=.3); plt.legend(fontsize=8)
plt.show()
